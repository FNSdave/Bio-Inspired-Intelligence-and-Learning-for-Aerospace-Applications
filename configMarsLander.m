function settings = configMarsLander()
%%% Mars-Lander RL — Configuration

%% 0) Run toggles

settings.rngSeed     = 42;
settings.trainAgent  = false;
settings.saveResults = true;
settings.runParallel = false;
settings.mainDevice  = "cpu";
settings.PlotsChoice        = "none";  % or "training-progress"
settings.plotEarlyTrajectory = false;
% --- guard to avoid accidental short episodes  ---
if ~isfield(settings,'maxStepsPerEpisode') || settings.maxStepsPerEpisode < 400
    settings.maxStepsPerEpisode = 400;   
end


%% 1) Timing
settings.dt                 = 0.1;     % [s]
settings.maxStepsPerEpisode = 400;
settings.total_max_episodes = 800;
settings.episodes_before_plot = 450;

%% 2) Vehicle & propulsion (Mars)
settings.m   = 5000;         % [kg]
settings.Isp = 225;          % [s] 
settings.g0  = 9.80665;      % [m/s^2]
settings.g   = 3.7114;       % [m/s^2] Mars
settings.fuel0 = 500;        % [kg] initial propellant for logging/plot

% Discrete thrusts (idle, main low/high, side ±)
settings.mainThrustLow   = 25e3;  % [N]
settings.mainThrustHigh  = 40e3;  % [N]
settings.sideThrust      = 10e3;  % [N]
settings.ActionSpace = [ 0, ...
                         settings.mainThrustLow, ...
                         settings.mainThrustHigh, ...
                         +settings.sideThrust, ...
                         -settings.sideThrust ];

%% 3) Touchdown envelope & reward weights
settings.v_limit    = 2.0;     % [m/s] acceptable |v| at touchdown
settings.fuelWeight = 5e-3;    % penalty per kg burned, was e-3

settings.rw.k_dist  = 3e-4;
settings.rw.k_speed = 5e-3;
settings.rw.k_side  = 1e-5;
settings.rw.k_main  = 1e-5;
settings.rw.R_exit  = -300;    % exit box, was 100
settings.rw.R_land  = +500;    % good landing, was 100 and 300
settings.rw.R_crash = -50;     % crash penalty base

%% 4) Box and terrain (use box_coordinates like Roversi/Langella)
settings.box_width       = 50;    % [m]
settings.box_height      = 50;    % [m]
settings.landingPadWidth = 10;    % [m]
settings.landingPadAlt   = 0;     % pad at 0

% 2×4 polygon for plots and bounds (L,R,top,bottom style)
settings.box_coordinates = [ -settings.box_width/2,  settings.box_width/2, ...
                              settings.box_width/2, -settings.box_width/2; ...
                             -settings.landingPadAlt, -settings.landingPadAlt, ...
                              settings.box_height - settings.landingPadAlt, ...
                              settings.box_height - settings.landingPadAlt ];

% Gentle slopes → flat pad 
padHalf = settings.landingPadWidth/2;
xL = settings.box_coordinates(1,1);
xR = settings.box_coordinates(1,2);

left_x  = linspace(xL, -padHalf, 4);
left_y  = [0, 0.15*rand*settings.box_height, 0.15*rand*settings.box_height, 0];
right_x = linspace(+padHalf, xR, 4);
right_y = [0, 0.15*rand*settings.box_height, 0.15*rand*settings.box_height, 0];
settings.ground_nodes = [left_x, right_x; left_y, right_y];

%% 5) Lander outline (plots)
scale = 0.6;
settings.shape_x = scale * [-5 -4 -2 -2  2  2  4  5  2  2 -2 -2 -5];
settings.shape_y = scale * [ 0  0  3  2  2  3  0  0  5  7  7  5  0];
settings.left_thrust_x     = scale * [-2 -4 -4 -2];
settings.left_thrust_y     = scale * [ 6  5  7  6];
settings.right_thrust_x    = scale * [ 2  4  4  2];
settings.right_thrust_y    = scale * [ 6  5  7  6];
settings.main_low_thrust_x = scale * [ 0  1 -1  0];
settings.main_low_thrust_y = scale * [ 0  2  2  0];
settings.main_high_thrust_x= scale * [ 0  1 -1  0];
settings.main_high_thrust_y= scale * [-1.5  2  2 -1.5];
end
