# Bio-Inspired-Intelligence-and-Learning-for-Aerospace-Applications
# Mars Lander Reinforcement Learning (DQN) - Davide Izzo (6311210)

This repository contains the MATLAB implementation of a Reinforcement Learning (RL) agent trained to autonomously land a vehicle on the surface of Mars using Deep Q-Networks (DQN). The agent learns to control the main and side thrusters to achieve a soft vertical landing while keeping in mind fuel consumption.

## Repository Structure

- `mainMarsLander.m`: Entry point for nominal training and simulation  
- `configMarsLander.m`: All environment and agent configuration settings  
- `advanceLander.m`, `initLanderState.m`: Environment dynamics and reset logic  
- `rewardMars.m`: Custom reward shaping for safe landing  
- `landingOutcome.m`: Checks and classifies landing success at the end of each episode  
- `TrajPlots_Mars.m`: Generates trajectory and velocity plots post-training  
- `RewardPlots_Mars.m`: Plots episode rewards and average reward evolution  
- `PropellantPlot_Mars.m`: Plots cumulative propellant consumption  
- `Sensitivity_*.m`: Scripts for one-at-a-time hyperparameter sensitivity analyses  
- `SimOut_Data/`: Automatically saved training statistics  
- `SimOut_Agents/`: Trained agent files
