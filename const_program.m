% returns the inlet air conditions
% at time "t", here simply given as constants
function [Ta_in, Wa_in] = const_program(t)
  % air inlet temperature, degC
  Ta_in = 60
  % air inlet moisture content, kg water/kg dry air
  Wa_in = 0.008
end
