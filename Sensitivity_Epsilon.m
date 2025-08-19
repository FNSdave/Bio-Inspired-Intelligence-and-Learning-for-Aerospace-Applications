%% -- Mars Lander: Sensitivity Analysis on ε -- %%
clc; close all; clearvars;

%% Parameter vector for epsilon values
epsilon_vec = [0.999, 0.9];  % exclude ε = 1 (already nominal)
episodeRewardCell = cell(1, length(epsilon_vec));
averageRewardCell = cell(1, length(epsilon_vec));
labels = strings(1, length(epsilon_vec));  % for legend

%% Loop over epsilon values
for iterator = 1:length(epsilon_vec)

    clearvars -except epsilon_vec episodeRewardCell averageRewardCell iterator labels

    % Fix seed and load settings
    rng(42)
    settings = configMarsLander();
    settings.trainAgent = true;
    settings.saveResults = true;
    settings.runParallel = false;

    % Save filenames
    tag = "epsilon";
    eps_str = strrep(num2str(epsilon_vec(iterator)), '.', '_');
    settings.data_filename  = "/SimOut_Data/trainingStats_" + tag + "_" + eps_str + ".mat";
    settings.agent_filename = "/SimOut_Agents/agent_"     + tag + "_" + eps_str + ".mat";
    settings.plot_reward_filename = "EpsilonSensitivity.jpg";

    % Adjust epsilon value
    settings.EpsilonValue = epsilon_vec(iterator);

    %% Environment
    ObservationInfo = rlNumericSpec([5 1]);
    ObservationInfo.Name = 'MarsLanderState';
    ActionInfo = rlFiniteSetSpec(settings.ActionSpace);
    ActionInfo.Name = 'MarsLanderThrust';

    ResetHandle = @() initLanderState(settings);
    StepHandle  = @(Action,LoggedSignals) advanceLander(Action,LoggedSignals,settings);
    env = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);

    %% Agent
    net = [
        featureInputLayer(5)
        fullyConnectedLayer(256, 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
        leakyReluLayer
        fullyConnectedLayer(256, 'WeightsInitializer', 'glorot', 'BiasInitializer', 'zeros')
        reluLayer
        fullyConnectedLayer(length(settings.ActionSpace))
    ];
    net = dlnetwork(net);
    critic = rlVectorQValueFunction(net, ObservationInfo, ActionInfo, UseDevice=settings.mainDevice);
    agent  = rlDQNAgent(critic);

    agent.AgentOptions.UseDoubleDQN = true;
    agent.AgentOptions.TargetSmoothFactor = 1;
    agent.AgentOptions.TargetUpdateFrequency = 5;
    agent.AgentOptions.MiniBatchSize = 128;
    agent.AgentOptions.ExperienceBufferLength = 200000;
    agent.AgentOptions.DiscountFactor = 0.99;
    agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
    agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;
    agent.AgentOptions.EpsilonGreedyExploration.Epsilon      = settings.EpsilonValue;
    agent.AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 0.05;
    agent.AgentOptions.EpsilonGreedyExploration.EpsilonMin   = 0.001;

    %% Train
    disp("Training with epsilon = " + settings.EpsilonValue)
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

    if settings.saveResults
        save(pwd + settings.data_filename, "trainingStats");
        save(pwd + settings.agent_filename, "agent");
    end

    % Store rewards
    episodeRewardCell{iterator} = trainingStats.EpisodeReward;
    averageRewardCell{iterator} = trainingStats.AverageReward;
    labels(iterator) = "ϵ = " + num2str(epsilon_vec(iterator));

end

%% Load nominal curve (ε = 1)
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
figure; hold on; grid on;
colors = lines(length(averageRewardCell));

for k = 1:length(averageRewardCell)
    plot(averageRewardCell{k}, 'LineWidth', 1.4, 'Color', colors(k,:));
end
yline(70, '--g', 'Reward Threshold', 'LineWidth', 1.3);
xlabel("Episode"); ylabel("Average Reward");
title("Epsilon Sensitivity");
legend(labels, 'Location', 'southwest');

% Save if requested
if settings.saveResults
    saveas(gcf, settings.plot_reward_filename);
end
