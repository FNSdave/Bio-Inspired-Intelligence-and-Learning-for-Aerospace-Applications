function [NextObs, Reward, IsDone, LoggedSignals] = advanceLander(Action, LoggedSignals, settings)
% One RL step (discrete actions → thrust; integrates 2-DOF; box-bounded)
% Plot-only fuel logging (kg). Dynamics remain with constant mass settings.m.

A  = settings.ActionSpace;
m  = settings.m;          % keep constant mass (no retrain path)
g  = settings.g;
dt = settings.dt;

Y   = LoggedSignals.State;     % [x; z; vx; vz; c]
vx  = Y(3);  vz = Y(4);

% Map action to thrust [Tx; Tz]
if     Action == A(1), T = [0; 0];
elseif Action == A(2), T = [0;  settings.mainThrustLow];
elseif Action == A(3), T = [0;  settings.mainThrustHigh];
elseif Action == A(4), T = [ settings.sideThrust; 0];
elseif Action == A(5), T = [-settings.sideThrust; 0];
else,  error('advanceLander: action not in ActionSpace');
end

% Dynamics (freeze after terminal)
if Y(5) == 0
    ax = T(1)/m;
    az = T(2)/m - g;
    dY = [vx; vz; ax; az; 0];
else
    dY = zeros(5,1);
end
Y = Y + dt*dY;

% Termination (impact vs box edges; no ceiling)
[Y_new, v_touchdown] = landingOutcome(Y, settings);

% -------- Propellant logging in kg (plot-only; does not change dynamics) --
% Ensure fields exist even if initLanderState wasn't updated yet
if ~isfield(LoggedSignals,'fuel_kg')
    fuel0 = isfield(settings,'fuel0') * settings.fuel0 + ~isfield(settings,'fuel0') * 500; % default 500 kg
    LoggedSignals.fuel_kg   = fuel0;
    LoggedSignals.fuelTrace = fuel0;
end
% ṁ = ||T||/(g0*Isp)  → kg/s
mdot    = norm(T) / (settings.g0 * settings.Isp);
burn_kg = mdot * dt;
LoggedSignals.fuel_kg   = max(LoggedSignals.fuel_kg - burn_kg, 0);
LoggedSignals.fuelTrace = [LoggedSignals.fuelTrace, LoggedSignals.fuel_kg];

% Reward (8 components + 9th = fuel)
R_vec  = rewardMars(Y_new, T, v_touchdown, burn_kg, settings);
Reward = sum(R_vec);

% Bookkeeping
LoggedSignals.State             = Y_new;
LoggedSignals.cumulativeState   = [LoggedSignals.cumulativeState, Y_new];
LoggedSignals.cumulativeThrust  = [LoggedSignals.cumulativeThrust, T];
LoggedSignals.cumulativeReward  = [LoggedSignals.cumulativeReward, R_vec];
LoggedSignals.velocityTouchdown = v_touchdown;

NextObs = LoggedSignals.State;
IsDone  = (Y_new(end) ~= 0);
end
