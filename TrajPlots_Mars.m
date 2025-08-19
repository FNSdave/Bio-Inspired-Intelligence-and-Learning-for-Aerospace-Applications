function TrajPlots_Mars(Y, T, settings)


t  = settings.dt * (0:size(Y,2)-1);
nT = min(size(T,2), max(0,size(Y,2)-1));
tT = settings.dt * (1:nT);

figure('Name','Mars-Lander Trajectory','Position',[200 80 900 800]);

% (1) Trajectory (speed colormap) with box fill
subplot(2,2,1); hold on;
fill([settings.ground_nodes(1,:) settings.box_coordinates(1,3) settings.box_coordinates(1,4)], ...
     [settings.ground_nodes(2,:) settings.box_coordinates(2,3) settings.box_coordinates(2,4)], ...
     [0 0 0], 'EdgeColor','none');
plot([-settings.landingPadWidth/2, settings.landingPadWidth/2], [0 0], 'g','LineWidth',2);
spd = hypot(Y(3,:), Y(4,:));
spd(~isfinite(spd)) = 0;   % keep color scale sane when velocities blow up
scatter(Y(1,:), Y(2,:), 10, spd, 'filled'); colorbar; colormap jet;
title('Trajectory (colored by speed)'); axis equal;
xlabel('x [m]'); ylabel('z [m]');
xlim([settings.box_coordinates(1,1) settings.box_coordinates(1,2)]);
ylim([min([settings.ground_nodes(2,:) 0])-5, max([settings.box_height max(Y(2,:))])+5]);
grid on;

% (2) Position vs time
subplot(2,2,2); hold on; grid on;
plot([t(1) t(end)], [0 0], 'k--','DisplayName','Pad z=0');
plot(t, Y(1,:), 'r','LineWidth',1.5,'DisplayName','x(t)');
plot(t, Y(2,:), 'b','LineWidth',1.5,'DisplayName','z(t)');
xlabel('t [s]'); ylabel('Position [m]'); legend('Location','northeast'); title('Position vs time'); xlim([t(1) t(end)]); pbaspect([2 1 1]);

% (3) Speed vs time + acceptable region
subplot(2,2,3); hold on; grid on;
plot(t, spd, 'k','LineWidth',1.2,'DisplayName','|v|');
plot([t(1) t(end)], [settings.v_limit settings.v_limit], 'g--','DisplayName','v_{limit}');
finiteSpd = spd(isfinite(spd));
if isempty(finiteSpd), finiteSpd = 0; 
end
yl = [0, max(max(finiteSpd)*1.1, settings.v_limit*1.3)];
fill([t(1) t(end) t(end) t(1)], [0 0 settings.v_limit settings.v_limit], [0 1 0], 'FaceAlpha',0.15, 'EdgeColor','none','DisplayName','OK region');
ylim(yl); xlabel('t [s]'); ylabel('|v| [m/s]'); title('Speed vs time'); legend('Location','best');

% (4) Thrust vs time
subplot(2,2,4); hold on; grid on;
stairs(tT, T(2,1:nT), 'b','DisplayName','Tz');
stairs(tT, T(1,1:nT), 'r','DisplayName','Tx');
xlabel('t [s]'); ylabel('Thrust [N]'); title('Thrust vs time'); legend('Location','best');
end
