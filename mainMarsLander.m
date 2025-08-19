%% -- Main code for MarsLander -- %%

%% Initialize code
clc
close all

% Fix seed for reproducibility
rng(42)

% Load configurations file
settings = configMarsLander();

%% Prepare computing setup (cap to 2 workers if parallel is requested)
nCores = feature('numcores');
p = gcp('nocreate');
if isempty(p) && settings.runParallel == true
    % Start a small pool
    pool = parpool(min(nCores, 2));
    disp(['Parallel pool running with cores: ', num2str(pool.NumWorkers)])
    
    try
        files = matlab.codetools.requiredFilesAndProducts(mfilename('fullpath'));
        addAttachedFiles(gcp, files);
    catch
        % fallback explicit attach
        addAttachedFiles(gcp, { ...
            'configMarsLander.m','initLanderState.m','advanceLander.m', ...
            'landingOutcome.m','rewardMars.m', ...
            'TrajPlots_Mars.m','RewardPlots_Mars.m','PropellantPlot_Mars.m' });
    end
else
    if settings.runParallel == true
        disp(['Parallel started with cores: ', num2str(p.NumWorkers)])
    else
        disp('Running single core')
    end
end

%% Create Observation and Action spaces
ObservationInfo = rlNumericSpec([5 1]); % Y = [x; z; vx; vz; c]
ObservationInfo.Name = 'MarsLanderState';
ObservationInfo.Description = 'x, z, vx, vz, c';

ActionInfo = rlFiniteSetSpec(settings.ActionSpace); % Discrete thrust actions
ActionInfo.Name = 'MarsLanderThrust';
ActionInfo.Description = 'Possible actions at each timestep';

%% Create environment using observation and action functions
ResetHandle = @() initLanderState(settings);
StepHandle  = @(Action,LoggedSignals) advanceLander(Action,LoggedSignals,settings);
env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);

%% Create DQN agent neural network 
net = [
    featureInputLayer(ObservationInfo.Dimension(1)) % input: 5
    fullyConnectedLayer(256, WeightsInitializer='glorot', BiasInitializer='zeros')
    leakyReluLayer
    fullyConnectedLayer(256, WeightsInitializer='glorot', BiasInitializer='zeros')
    reluLayer
    fullyConnectedLayer(length(ActionInfo.Elements)) % output: #actions
];
net = dlnetwork(net);

%% Create the agent based on the DQN neural network
if ~isfield(settings,'mainDevice'); settings.mainDevice = "cpu"; end
critic = rlVectorQValueFunction(net,ObservationInfo,ActionInfo, UseDevice=settings.mainDevice);

agent = rlDQNAgent(critic);
reset(agent);   % clears the replay buffer (weights stay as initialized)
agent.AgentOptions.UseDoubleDQN = true;
agent.AgentOptions.TargetSmoothFactor = 1; %was 1, 1e-3
agent.AgentOptions.TargetUpdateFrequency = 5; %was 5, 1
agent.AgentOptions.MiniBatchSize = 128;
agent.AgentOptions.ExperienceBufferLength = 200000;
agent.AgentOptions.EpsilonGreedyExploration.Epsilon = 1; %eas 0.99
agent.AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 0.05; %was 0.05, 1e-3
agent.AgentOptions.EpsilonGreedyExploration.EpsilonMin = 0.001; %was 0.001, 0.05
agent.AgentOptions.DiscountFactor = 0.99;
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3; %was 1e-3
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

%% Train the agent
settings.resultType = "training";
if settings.trainAgent == true
    if isfield(settings,'plotBadTrajectory') && settings.plotBadTrajectory == true
        % Short preview run (no saving)
        trainOpts = rlTrainingOptions( ...
            MaxEpisodes              = settings.episodes_before_plot, ...
            MaxStepsPerEpisode       = settings.maxStepsPerEpisode, ...
            Verbose                  = false, ...
            Plots                    = "none", ...
            StopTrainingCriteria     = "EpisodeCount", ...
            StopTrainingValue        = settings.episodes_before_plot, ...
            UseParallel              = false );
        trainingStats = train(agent,env,trainOpts); 

    else
        % Full run 
        trainOpts = rlTrainingOptions( ...
            MaxEpisodes                  = settings.total_max_episodes, ...
            MaxStepsPerEpisode           = settings.maxStepsPerEpisode, ...
            Verbose                      = false, ...
            Plots                        = "training-progress", ...
            StopTrainingCriteria         = "AverageReward", ...
            ScoreAveragingWindowLength   = 50, ...
            StopTrainingValue            = 70, ...
            UseParallel                  = settings.runParallel );

        trainingStats = train(agent,env,trainOpts); %#ok<NASGU>

        % Save 
        if ~exist(fullfile(pwd,"SimOut_Data"),"dir");   mkdir(fullfile(pwd,"SimOut_Data"));   end
        if ~exist(fullfile(pwd,"SimOut_Agents"),"dir"); mkdir(fullfile(pwd,"SimOut_Agents")); end
        save(fullfile(pwd,"SimOut_Data","trainingStats.mat"), 'trainingStats');
        save(fullfile(pwd,"SimOut_Agents","agentRef.mat"),   'agent');
    end


    if settings.runParallel == false
        % Use the environment logs from the last training episode
        Y_tot = env.LoggedSignals.cumulativeState;    % 5×N
        T_tot = env.LoggedSignals.cumulativeThrust;   % 2×N
        R_tot = env.LoggedSignals.cumulativeReward;   % 9×N

        TrajPlots_Mars(Y_tot, T_tot, settings);
        RewardPlots_Mars(trainingStats, settings);        
        PropellantPlot_Mars(env.LoggedSignals.fuelTrace, settings);

        % Console summary
        disp("Cumulative reward: " + num2str(sum(sum(R_tot))))
        disp("Penalty proportional distance: " + num2str(sum(R_tot(1, :))))
        disp("Penalty proportional speed: " + num2str(sum(R_tot(2, :))))
        disp("Penalty side engines: " + num2str(sum(R_tot(3, :))))
        disp("Penalty main engine: " + num2str(sum(R_tot(4, :))))
        disp("Penalty exit boundaries: " + num2str(sum(R_tot(5, :))))
        disp("Penalty crash outside: " + num2str(sum(R_tot(6, :))))
        disp("Penalty crash inside landing pad: " + num2str(sum(R_tot(7, :))))
        disp("Reward successful landing " + num2str(sum(R_tot(8, :))))
        disp("Touchdown vx " + num2str(env.LoggedSignals.velocityTouchdown(1)))
        disp("Touchdown vz " + num2str(env.LoggedSignals.velocityTouchdown(2)))
        disp("Total number of steps: " + num2str(size(Y_tot, 2)))
        disp("Total landing duration in simulation time [s]: " + num2str(size(Y_tot, 2)*settings.dt))
    end

    disp("Training terminated")
    disp("-----------------")
else
    S = load(fullfile(pwd,"SimOut_Agents","agentRef.mat"),'agent');   agent = S.agent;
    if isfile(fullfile(pwd,"SimOut_Data","trainingStats.mat"))
        S2 = load(fullfile(pwd,"SimOut_Data","trainingStats.mat"),'trainingStats');
        trainingStats = S2.trainingStats;
    end
end
%% (Optional) Per-step reward plot for the just-finished training episode
% (This replaces plotting from trainingStats.)
if ~isfield(settings,'plotBadTrajectory') || settings.plotBadTrajectory == false
    RewardPlots_Mars(trainingStats, settings);
end

%% Simulate the trained agent once (fresh episode) and plot again
rng(0)
simOpts = rlSimulationOptions(MaxSteps=settings.maxStepsPerEpisode, NumSimulations=1);
experience = sim(env,agent,simOpts);

% Use environment logs for this simulation episode
R_hist = env.LoggedSignals.cumulativeReward;   % 9×N reward components
Y_log  = env.LoggedSignals.cumulativeState;    % 5×N states
T_log  = env.LoggedSignals.cumulativeThrust;   % 2×N thrust history

% Or reconstruct thrusts from the experience actions (either is fine)
Y_sim = squeeze(experience(1).Observation.MarsLanderState.Data(:,1,:));  % 5×N
act   = squeeze(experience(1).Action.MarsLanderThrust.Data(1,1,:)).';

Tx = zeros(1,numel(act));  Tz = zeros(1,numel(act));
Tx(act==settings.ActionSpace(4)) = settings.ActionSpace(4);
Tx(act==settings.ActionSpace(5)) = settings.ActionSpace(5);
Tz(act==settings.ActionSpace(2)) = settings.ActionSpace(2);
Tz(act==settings.ActionSpace(3)) = settings.ActionSpace(3);
T_sim = [0, Tx; 0, Tz];

% Plots for the simulation episode 
TrajPlots_Mars(Y_sim, T_sim, settings);
RewardPlots_Mars(trainingStats, settings);
PropellantPlot_Mars(env.LoggedSignals.fuelTrace, settings);
