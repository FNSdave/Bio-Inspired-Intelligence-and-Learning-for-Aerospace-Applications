function [Y_new, v_touchdown] = landingOutcome(Y, settings)
% Impact vs terrain and box edges

x = Y(1);  z = Y(2);  vx = Y(3);  vz = Y(4);

% Terrain height at x
z_impact = interp1(settings.ground_nodes(1,:), settings.ground_nodes(2,:), x, 'linear', 'extrap');

% Box limits from polygon (left/right/top; bottom=pad)
outLeft  = x < settings.box_coordinates(1,1);
outRight = x > settings.box_coordinates(1,2);
outTop   = z > settings.box_coordinates(2,3);   % top of box

if outLeft || outRight || outTop
    c = 2;                           % exit boundaries
    v_touchdown = [inf, inf];
    Y_new = [x; z; vx; vz; c];
    return
end

% Ground contact
if z < z_impact
    z = z_impact;                    % snap to ground
    c = 1;                           % terminal (contact)
    v_touchdown = [vx, vz];
else
    c = 0;
    v_touchdown = [0, 0];
end

Y_new = [x; z; vx; vz; c];
end
