% returns the inlet air conditions
% at time "t", here based on lopez's
% program of increased temperature, 
% constant humidity
function [Ta_in, Wa_in] = lopez_program(t)
  hours_elapsed = t/double(3600);
  if hours_elapsed < 20
    Ta_in = 60;
    Wa_in = 0.008;
  elseif hours_elapsed > 20 && hours_elapsed < 24
    Ta_in = 65;
    Wa_in = 0.008;
  else
    Ta_in = 80;
    Wa_in = 0.008;
  end
end
