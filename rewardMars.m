function Reward_vec = rewardMars(Y, T, v_touchdown, burn_kg, settings)
x = Y(1); z = Y(2); vx = Y(3); vz = Y(4); c = Y(5);
spd = hypot(vx,vz);

% --- shaping ---
R_proportional_dist  = -settings.rw.k_dist  * norm([x, z]);
R_proportional_speed = -settings.rw.k_speed * spd;


% --- initialize conditionals to zero  ---
R_side_engines = 0;
R_main_engine  = 0;
R_exit_boundaries = 0;
R_crash_outside_landing_pad = 0;
R_crash_inside_landing_pad  = 0;
R_landing = 0;

% --- engine penalties  ---
if T(1) ~= 0
    R_side_engines = -0.03;
end
if T(2) ~= 0
    R_main_engine = -0.3;
end

% --- exit/landing/crash ---
if c == 2
    R_exit_boundaries = settings.rw.R_exit;
elseif c == 1
    onPad = (abs(x) <= settings.landingPadWidth/2);
    if onPad && spd < settings.v_limit
        R_landing = settings.rw.R_land;
    elseif onPad
        R_crash_inside_landing_pad = -spd;
    else
        R_crash_outside_landing_pad = settings.rw.R_crash - spd;
    end
end

% --- fuel term ---
R_fuel = -settings.fuelWeight * burn_kg;

% --- pack vector  ---
Reward_vec = [ R_proportional_dist;
               R_proportional_speed;
               R_side_engines;
               R_main_engine;
               R_exit_boundaries;
               R_crash_outside_landing_pad;
               R_crash_inside_landing_pad;
               R_landing;
               R_fuel ];
end
