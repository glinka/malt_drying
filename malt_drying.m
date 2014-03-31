function [ouput_vars] = malt_drying(input_vars)

end

% latent heat of water vaporization calc
% from 1510 of Lopez et al.
function [lhwv] = L(Ta)
lhwv = 2500.6 - 2.3643956*Ta;
end

% air enthalpy calc
% from 1510 of Lopez et al.
function [ae] = i(Ca, Ta, wa, La, Cv)
ae = Ca*Ta + wa*(La+Cv*Ta);
end

% saturated water vapor pressure calc
% from 1510 of Lopez et al.
function [pressure] = P(Ta)
if((Ta + 273.15) < 60)
    pressure = exp(14.293  - 5291/(Ta + 273.15))/(3.2917 - 0.01527*(Ta + 273.15) + 2.52E-5*(Ta + 273.15))^2;
else
    Disp('Ta larger than 60 C')
    pressure = exp(14.293  - 5291/(Ta + 273.15))/(3.2917 - 0.01527*(Ta + 273.15) + 2.52E-5*(Ta + 273.15))^2;
end

% GAB contants, A, B, and C calc
% from 1510 of Lopez et al.
function [a] = A(T)
a = 0.01183*exp(469.017/T);
end

function [b] = B(T)
b = exp(943.854/T);
end

function [c] = C(T)
c = exp(-28.639/T);
end

function 
