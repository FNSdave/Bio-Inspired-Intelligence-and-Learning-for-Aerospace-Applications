%% -- Main script for Mars Lander Sensitivity Analysis: Learning Rate -- %%
clc; close all; clear;

%% Define values to test for learning rate (alpha)
alpha_values = [1e-4, 1e-2];  % Nominal 1e-3 is fixed in config

%% Prepare result containers
episodeRewardCell = cell(1, length(alpha_values));
averageRewardCell = cell(1, length(alpha_values));

%% Loop over different alpha values
for i = 1:length(alpha_values)
    close all; clear agent;

    % Fix seed for reproducibility
    rng(42)

    % Load configuration
    settings = configMarsLander();

    % Set unique file naming for this run
    settings.plot_reward_filename = "alpha.jpg";
    settings.plot_filename        = "alpha_" + num2str(alpha_values(i)) + ".jpg";
    settings.data_filename        = "trainingStats_alpha_" + num2str(alpha_values(i)) + ".mat";
    settings.agent_filename       = "agent_alpha_" + num2str(alpha_values(i)) + ".mat";

    % Create Observation and Action spaces
    ObservationInfo = rlNumericSpec([5 1]);
    ObservationInfo.Name = 'MarsLanderState';

    ActionInfo = rlFiniteSetSpec(settings.ActionSpace);
    ActionInfo.Name = 'MarsLanderThrust';

    % Environment
    ResetHandle = @() initLanderState(settings);
    StepHandle  = @(Action, LoggedSignals) advanceLander(Action, LoggedSignals, settings);
    env = rlFunctionEnv(ObservationInfo, ActionInfo, StepHandle, ResetHandle);

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

    % Define critic and agent
    critic = rlVectorQValueFunction(net, ObservationInfo, ActionInfo);
    agent = rlDQNAgent(critic);
    agent.AgentOptions.UseDoubleDQN = true;
    agent.AgentOptions.TargetSmoothFactor = 1;
    agent.AgentOptions.TargetUpdateFrequency = 5;
    agent.AgentOptions.MiniBatchSize = 128;
    agent.AgentOptions.ExperienceBufferLength = 200000;
    agent.AgentOptions.EpsilonGreedyExploration.Epsilon      = 1;
    agent.AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 0.05;
    agent.AgentOptions.EpsilonGreedyExploration.EpsilonMin   = 0.001;
    agent.AgentOptions.DiscountFactor = 0.99;
    agent.AgentOptions.CriticOptimizerOptions.LearnRate = alpha_values(i);
    agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

    disp("====================================================")
    disp("Training with alpha = " + num2str(alpha_values(i)))
    disp("====================================================")

    % Train the agent
    settings.resultType = "training";
    trainOpts = rlTrainingOptions(...
        MaxEpisodes = settings.total_max_episodes, ...
        MaxStepsPerEpisode = settings.maxStepsPerEpisode, ...
        ScoreAveragingWindowLength = 50, ...
        StopTrainingCriteria = "AverageReward", ...
        StopTrainingValue = 70, ...
        Plots = "none", ...
        UseParallel = settings.runParallel, ...
        Verbose = false);

    trainingStats = train(agent, env, trainOpts);

    % Save if enabled
    if settings.saveResults
        if ~exist("SimOut_Data", "dir"); mkdir("SimOut_Data"); end
        if ~exist("SimOut_Agents", "dir"); mkdir("SimOut_Agents"); end
        save(fullfile("SimOut_Data", settings.data_filename), 'trainingStats');
        save(fullfile("SimOut_Agents", settings.agent_filename), 'agent');
    end

    % Store reward time series for final reward comparison plot
    episodeRewardCell{1, i} = trainingStats.EpisodeReward;
    averageRewardCell{1, i} = trainingStats.AverageReward;
end

%% Plot all reward curves (episodes and average)
figure;
hold on; grid on;
colors = lines(length(alpha_values));
for i = 1:length(alpha_values)
    plot(averageRewardCell{1,i}, 'LineWidth', 1.4, 'Color', colors(i,:));
end
legend(compose('\\alpha = %.0e', alpha_values), 'Location', 'northwest');
xlabel('Episode'); ylabel('Average Reward');
title('Learning Rate Sensitivity');
saveas(gcf, fullfile("SimOut_Data", settings.plot_reward_filename));
