%% -- Sensitivity Analysis: MiniBatchSize (MBS) -- %%
clc; close all; clearvars;

%% Values to test
valuesToTest = [64, 256];  % MBS values
nominalValue = 128;        % Include in plot only, not retrain
labels = ["MBS = 64", "MBS = 256"];
tag = "MBS";

%% Preallocate results
nCases = numel(valuesToTest);
episodeRewardCell = cell(1, nCases);
averageRewardCell = cell(1, nCases);

%% Loop on each MBS value
for i = 1:nCases

    % Clear agent and close plots
    clear agent
    close all

    % Fix seed and load settings
    rng(42)
    settings = configMarsLander();

    % Override key parameters
    settings.trainAgent = true;
    settings.runParallel = false;
    settings.saveResults = true;

    % Create observation and action space
    ObservationInfo = rlNumericSpec([5 1]);
    ObservationInfo.Name = 'MarsLanderState';
    ActionInfo = rlFiniteSetSpec(settings.ActionSpace);
    ActionInfo.Name = 'MarsLanderThrust';

    % Create environment
    ResetHandle = @() initLanderState(settings);
    StepHandle  = @(Action, LoggedSignals) advanceLander(Action, LoggedSignals, settings);
    env = rlFunctionEnv(ObservationInfo, ActionInfo, StepHandle, ResetHandle);

    % Create DQN agent network (fixed 256-256)
    net = [
        featureInputLayer(5)
        fullyConnectedLayer(256, 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
        leakyReluLayer
        fullyConnectedLayer(256, 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
        reluLayer
        fullyConnectedLayer(numel(settings.ActionSpace))
    ];
    net = dlnetwork(net);

    % Create critic and agent
    critic = rlVectorQValueFunction(net, ObservationInfo, ActionInfo, UseDevice=settings.mainDevice);
    agent = rlDQNAgent(critic);
    reset(agent);

    % Set agent options
    agent.AgentOptions.UseDoubleDQN = true;
    agent.AgentOptions.TargetSmoothFactor = 1;
    agent.AgentOptions.TargetUpdateFrequency = 5;
    agent.AgentOptions.MiniBatchSize = valuesToTest(i);  % <<--- key change
    agent.AgentOptions.ExperienceBufferLength = 200000;
    agent.AgentOptions.EpsilonGreedyExploration.Epsilon = 1;
    agent.AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 0.05;
    agent.AgentOptions.EpsilonGreedyExploration.EpsilonMin = 0.001;
    agent.AgentOptions.DiscountFactor = 0.99;
    agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
    agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

    disp("Training with MiniBatchSize = " + num2str(valuesToTest(i)))

    % Train the agent
    trainOpts = rlTrainingOptions( ...
        MaxEpisodes = settings.total_max_episodes, ...
        MaxStepsPerEpisode = settings.maxStepsPerEpisode, ...
        ScoreAveragingWindowLength = 50, ...
        StopTrainingCriteria = "AverageReward", ...
        StopTrainingValue = 70, ...
        UseParallel = settings.runParallel, ...
        Verbose = false, ...
        Plots = settings.PlotsChoice);

    trainingStats = train(agent, env, trainOpts);

    % Save files using Roversi-like tags
    save(fullfile(pwd, "SimOut_Data", "trainingStats_" + tag + "_" + num2str(valuesToTest(i)) + ".mat"), "trainingStats");
    save(fullfile(pwd, "SimOut_Agents", "agent_" + tag + "_" + num2str(valuesToTest(i)) + ".mat"), "agent");

    % Store rewards
    episodeRewardCell{i} = trainingStats.EpisodeReward;
    averageRewardCell{i} = trainingStats.AverageReward;

end

%% Load nominal curve (do not retrain)
nominalFile = fullfile(pwd, "SimOut_Data", "trainingStats.mat");
if isfile(nominalFile)
    S = load(nominalFile);
    averageRewardNominal = S.trainingStats.AverageReward;
    averageRewardCell{end+1} = averageRewardNominal;
    labels(end+1) = "Nominal";
else
    warning("Nominal trainingStats.mat not found. Nominal curve will be excluded.");
end

%% Plot sensitivity
figure;
hold on; grid on;
for i = 1:numel(averageRewardCell)
    plot(averageRewardCell{i}, 'LineWidth', 1.4);
end
yline(70, '--g', 'Reward Threshold', 'LineWidth', 1.3);
xlabel('Episode'); ylabel('Average Reward');
title('MiniBatchSize Sensitivity');
legend(labels, 'Location', 'southeast');
