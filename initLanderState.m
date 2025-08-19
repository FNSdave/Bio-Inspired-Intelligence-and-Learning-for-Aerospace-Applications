function [InitialObservation, LoggedSignals] = initLanderState(settings)
% Reset using the box polygon  

% Start near top-right inside the box (x along +, z near top)
x0  = 0.75 * settings.box_coordinates(1,2);
z0  = 0.75 * settings.box_coordinates(2,3); %was 0.9 and 0.75
vx0 = 0;
vz0 = -0.5;
c0  = 0;

% State and logging 
LoggedSignals.State            = [x0; z0; vx0; vz0; c0];
LoggedSignals.cumulativeState  = [x0; z0; vx0; vz0; c0];      % 5×N
LoggedSignals.cumulativeReward = zeros(9,1);  % 9 components (incl. fuel)
LoggedSignals.cumulativeThrust = [];
LoggedSignals.velocityTouchdown= [0, 0];      % will be overwritten
LoggedSignals.fuel_kg   = settings.fuel0;
LoggedSignals.fuelTrace = settings.fuel0;   % store remaining fuel (kg)

InitialObservation = LoggedSignals.State;
end
