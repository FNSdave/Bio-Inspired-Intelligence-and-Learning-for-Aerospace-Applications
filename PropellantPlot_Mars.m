function PropellantPlot_Mars(fuelTrace_kg, settings)
t = settings.dt * (0:numel(fuelTrace_kg)-1);
remain = fuelTrace_kg;
used   = settings.fuel0 - remain;
figure('Name','Mars-Lander Propellant');
yyaxis left;  plot(t, remain, 'LineWidth',1.6); ylabel('Fuel remaining [kg]');
yyaxis right; plot(t, 100*remain/settings.fuel0, '--', 'LineWidth',1.2); ylabel('Fuel remaining [%]');
xlabel('t [s]'); grid on; title('Fuel usage over time');
end
