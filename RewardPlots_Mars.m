function RewardPlots_Mars(trainingStats, settings)


% Expects: trainingStats with fields EpisodeReward, AverageReward (MATLAB RL Toolbox)

% Extract series
episodeReward = trainingStats.EpisodeReward(:);
if isfield(trainingStats,'AverageReward') && ~isempty(trainingStats.AverageReward)
    averageReward = trainingStats.AverageReward(:);
else
    % Fallback: rolling mean over last 50
    w = min(50, numel(episodeReward));
    averageReward = movmean(episodeReward,[w-1 0]);
end
t = 0:numel(episodeReward)-1;

% Plot
figure('Name','Rewards in time','Position',[300 70 800 400]);
hold on; grid on;
plot(t, episodeReward, 'Color',[0.3010 0.7450 0.9330], 'LineWidth', 1.5, ...
    'DisplayName','Episode Reward');
plot(t, averageReward, 'Color',[0 0.4470 0.7410], 'LineWidth', 2, ...
    'DisplayName','Average reward (50 episodes)');
xlabel('Episodes [-]'); ylabel('Reward values [-]');
legend('Location','northeast');
xlim([t(1) t(end)]);
if isfield(settings,'yRewardLim') && ~isempty(settings.yRewardLim)
    ylim(settings.yRewardLim);
set(gca,'FontSize',12);
hold off;

% Save if requested
if isfield(settings,'saveResults') && settings.saveResults
    outDir = fullfile(pwd,'SimOut_Media');
    if ~exist(outDir,'dir'); mkdir(outDir); end
    saveas(gcf, fullfile(outDir,'training_rewards.jpg'));
end
end
