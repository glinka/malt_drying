function [ouput_vars] = malt_drying(input_vars)
% specify constants
% ??from user input??
% malt radius (m)
r = .005;
% specific heat values are assumed constant,
% and were taken from engineeringtoolbox
% at around 30 Celsius
% specific heat of water (kJ/(kg K)
Cp = 4.181;
% specific heat of water vapor  (kJ/(kg K))
Cpw = 1.87;
% specific heat of drying air (kJ/(kg K))
Ca = 1.006;
% density of air (kg/m^3)
RHOg = 1.225;
% density of dry hops
% ??value??
% taken from engineeringtoolbox 
RHOm = 561;
% velocity of drying air (m/s)
Vg = 0.25;
% air flow rate (kg/ (m^2 s))
G = Vg*RHOg
% atmospheric pressure (kPa)
P = 101.325;
% initial malt moisture content (kg water/kg dry matter)
Mo = 0.72;
% inlet air temperature (C)
Ta_in = 60;
% inlet air moisture (kg water/kg dry air)
Wa_in = 0.008;
% final time (sec)
t_final = 26*60*60;
% timestep (sec)
dt = 0.000001;
% number of timesteps
ntimes = t_final/dt;
% height of bed (m)
H = 1;
% whatever the bed height
% use a constant number of
% discretized levels
nZs = 10000;
% incremental height (m)
dZ = H/nZs;
% set aside space to save
% results, add one to account
% for t = 0, z = 0
time_save_interval = 10000*60;
z_save_interval = 100;
Tm_save = zeros(ntimes/time_save_interval, nZs/z_save_interval);
M_save = zeros(ntimes/time_save_interval, nZs/z_save_interval);
% begin simulation procedure
% initial malt temperature (C)
Tmo = 16;
Tms = Tmo*ones(nZs/z_save_interval + 1);
% initial malt moisture contents (kg water/kg dry matter)
Ms = 0.72*ones(nZs/z_save_interval + 1);
% initial air temperature in bed, assumed to be same as initial malt temperature (C)
Tas = Tmo*ones(nZs/z_save_interval + 1);
% initial air moisture content in bed (kg water/kg dry air)
Was = 0.008*ones(nZs/z_save_interval + 1);
Tas(1) = Ta_in;
Was(1) = Wa_in;
Aw = 0.5
for tc = 1:ntimes
    t = tc*dt;
    
    % calculate new values of M and Tm
    % for z = 0, then loop over rest
    Ta = Ta_in
    Wad = Wad_in;
    Tm = Tms(1);
    M = Ms(1);
    %    K = k(deff(Ta), r)
    K = k(Ta)
    Me = me(Wad, Ta)
    %    Me = me(GABa(Ta), GABb(Ta), GABc(Ta), Aw)
    dM = dm(k, M, Me, dt)
    % M = M + dM;
    Cm = cm(M)
    %    La = la(Ta)
    Lv = lv(Ta)
    % Lm = lm(La, M)
    Tg_f = tg(Ta, Wad, dZ, rhod, G, dt, Cpd, dM, Cpw, Tg, dWad, HcAs, Lv)
    dTm = dtm(a(Ta, Tm), RHOm, dM, dt, y(Lm, Cv, Ta, Cw, Tm), hcv(G), dZ, G, e(Ca, Cv, Wa, RHOm, dZ, G, dM, dt), f(Cv, Ta, La, Cw, Tm), b(Cm, Cw, M), Cw)
    Tm = Tm + dTm;
    Tms(1) = Tm;
    Ms(1) = M;
    pause = input('press enter to continue');
    for Z_counter = 1:nZs
        % create temp values from array
        
        % note that due to the unfortunate
        % fact that MATLAB indices begin at
        % (1), when
        % note that Z_counter indexes the
        % current level that we are calculating
        % at, and this WILL NOT correspond to
        % the proper indices in our arrays
        % this leads to the foolish/confusing looking
        % indices below
        
        Ta = Tas(Z_counter);
        Wa = Was(Z_counter)
        Tm = Tms(Z_counter+1)
        M = Ms(Z_counter+1);
        Z = Z_counter*dZ;
        K = k(deff(Ta), r);
        Me = me(GABa(Tm), GABb(Tm), GABc(Tm), Aw)
	dM = m(M, K, dt, Me) - M
        M = M + dM;
        Cm = cm(M);
        La = la(Ta);
        Lm = lm(La, M);
        dTm = dtm(a(Ta, Tm), RHOm, dM, dt, y(Lm, Cv, Ta, Cw, Tm), hcv(G), dZ, G, e(Ca, Cv, Wa, RHOm, dZ, G, dM, dt), f(Cv, Ta, La, Cw, Tm), b(Cm, Cw, M), Cw)
        Tm = Tm + dTm;
        dTa = dta(RHOm, dZ, G, dt, dM, Cv, Ta, La, Cw, Tm, dTm, Ca, M, Wa)
        dWa = dwa(RHOm, dZ, G, dM, dt)
        RH = rh(pw(Wa+dWa, P), ps(Ta+dTa))
	pause = input('press enter to continue');
        % continue updating until RH < 98%
        while RH > 98
            dTm = dtm(a(Ta, Tm), RHOm, dM, dt, y(Lm, Cv, Ta, Cw, Tm), hcv(G), dZ, G, e(Ca, Cv, Wa, RHOm, dZ, G, dM, dt), f(Cv, Ta, La, Cw, Tm), b(Cm, Cw, M), Cw);
            Tm = Tm + dTm;
            dTa = dta(RHOm, dZ, G, dt, dM, Cv, Ta, La, Cw, Tm, dTm, Ca, M, Wa);
            dWa = dwa(RHOm, dZ, G, dM, dt);
            RH = rh(pw(Wa+dWa, P), ps(Ta+dTa));
            disp(RH)
        end
        % update arrays
        Tas(Z_counter + 1) = Ta + dTa;
        Was(Z_counter + 1) = Wa + dWa;
        Tms(Z_counter + 1) = Tm;
        Ms(Z_counter + 1) = M;
        
        M_save(tc, Z_counter) = M;
        Tm_save(tc, Z_counter) = Tm;
    end
    %!! not actually updating contraction of bed !!
    disp(M - Mo)
    disp('Contraction is:');
    disp(s(Mo, M));
    %!! here is where air recirculation
    % calcs would be made !!
end
times = 0:ntimes;
plot(times, Ms(:, 50));
plot(times, Tms(:, 50));
end


% the following equations, to the end of
% the file, are those used in the above
% solution process, as specified in
% Lopez et a.

% latent heat of water vaporization calc
% from 1510 of Lopez et al.
% previously La
function [Lv] = lv(Ta)
  Lv = 2500.6 - 2.3643956*Ta;
end

% air enthalpy calc
% from 1510 of Lopez et al.
function [Ia] = ia(Ca, Ta, wa, La, Cv)
  Ia = Ca*Ta + wa*(La+Cv*Ta);
end

% saturated water vapor pressure calc
% from 1510 of Lopez et al.
function [Ps] = ps(Ta)
if((Ta + 273.15) < 60)
  % need to multiply by 100 to convert to (kPa)
    Ps = 100*exp(14.293  - 5291/(Ta + 273.15))/(3.2917 - 0.01527*(Ta + 273.15) + 2.54E-5*(Ta + 273.15)^2);
else
    % disp('Ta larger than 60 C')
    Ps = 100*exp(14.293  - 5291/(Ta + 273.15))/(3.2917 - 0.01527*(Ta + 273.15) + 2.54E-5*(Ta + 273.15)^2);
end
end

% GAB contants, A, B, and C calc
% from 1510 of Lopez et al.
function [A] = GABa(Tm)
  A = 0.01183*exp(469.017/(Tm + 273.15));
end

function [B] = GABb(Tm)
  B = exp(943.854/(Tm + 273.15));
end

function [C] = GABc(Tm)
  C = exp(-28.639/(Tm + 273.15));
end

% drying constant value
% from 1510 of Lopez et al.
% function [K] = k(Deff, r)
% K = Deff/r^2;
function [K] = k(Ta)
  K =139.3*exp(-4426/(Ta+273));
end

% effective diffusivity
% from 1510 of Lopez et al.
function [Deff] = deff(Ta)
  Deff = 0.41036*exp(-5108.4454/(Ta + 273.15));
end

% air moisture content
% from 1509 of Lopez et al.
function [Wa] = wa(RH, Ps, P)
  Wa = 0.622*(RH*Ps)/(P-RH*Ps);
end

% average moisture content of malt at equilibrium
% from 1509 of Lopez et al.
% function [Me] = me(A, B, C, Aw)
% Me = A*B*C*Aw/((1-C*Aw)*(1+(B-1)*C*Aw));

% from ocallagan

function [Me] = me(Wad, Ta)
% Moisture content, wet basis
% Wad = Wa/(1 + Wa)
  Me = 7040*sqrt(Wad)/(1.8*Ta + 32)^2 + 0.06015
end

% bed contraction coefficient
% ??units??
% from 1509 of Lopez et al.
function [S] = s(Mo, M)
  S = 25.2086*(1-exp(-0.04238*(Mo-M)));
end

% convective heat transfer coefficient
% from 1509 of Lopez et al.
function [Hcv] = hcv(G)
  Hcv = 49.32E-3*G^0.6;
end

% latent heat of malt water evaporation
% from 1509 of Lopez et al.
function [Lm] = lm(La, M)
  Lm = La*(1+0.5904*exp(-0.1367*M));
end

% malt specific heat
% from 1509 of Lopez et al.
function [Cpd] = cpd(M)
  Cpd = 1.651 + 0.004116*M;
end

% constants appearing in malt grain temperature
% change equation on page 1508
% from 1509 of Lopez et al.
function [A] = a(Ta, Tm)
  A = 2*(Ta + Tm);
end

function [B] = b(Cm, Cw, M)
  B = Cm + Cw*M;
end

function [E] = e(Ca, Cv, Wa, RHOm, dZ, G, dM, dt)
  E = Ca + Cv*(Wa - (RHOm*dZ*dM)/(G*dt));
end

function [F] = f(Cv, Ta, La, Cw, Tm)
  F = Cv*Ta + La - Cw*Tm;
end

function [Y] = y(Lm, Cv, Ta, Cw, Tm)
  Y = Lm + Cv*Ta - Cw*Tm;
end

% malt grain temperature change
% from 1508 of Lopez et al. (eqn 16)
function [dTm] = dtm(A, RHOm, dM, dt, Y, Hcv, dZ, G, E, F, B, Cw)
  numerator = A + (RHOm*dM/dt)*(2*Y/Hcv + dZ*F/(G*E))
  denom = 1 + (RHOm/dt)*(2*B/Hcv + dZ*(B+Cw*dM)/(G*E))
  dTm = numerator/denom;
end

% air temperature change in elemental layer
% from 1508 of Lopez et al.
function [dTa] = dta(RHOm, dZ, G, dt, dM, Cv, Ta, La, Cw, Tm, dTm, Ca, M, Wa)
  numerator = (RHOm*dZ/(G*dt))*(dM*(Cv*Ta + La - Cw*Tm) - dTm*(Ca + Cw*(M + dM)))
  denom = Ca + Cv*(Wa - RHOm*dZ*dM/(G*dt))
  dTa = numerator/denom;
end

% air moisture change
% from 1506 of Lopez et al.
% function [dWa] = dwa(RHOm, dZ, G, dM, dt)
% dWa = -RHOm*dZ*dM/(G*dt);

% from ocallagan
function [dWa] = dwa(rhod, dZ, dM, G, dt)
  dWa = -dM*rhod*dZ/(G*dt)
end

% from ocallagan
function [dM] = dm(k, M, Me, dt)
  dM = -k*(M - Me)*dt/(1 + k*dt/2)
end

% from ocallagan
function [Tg_future] = tg(Ta, Wad, dZ, rhod, G, dt, Cpd, dM, Cpw, Tg, dWad, HcAs, Lv)
  F = -dZ*rhod/(G*dt)
  D = 2*rhod/(HcAs*dt)
  term1 = Ta*(1005 + 1820*Wad) + Tg*(-F*(Cpd+M*Cpw)) - 2501000*dWad
  term2 = (1005 + 1820*(Wad+dWad))*(Ta + D*dM*Lv + D*Tg*(Cpd + M))/(1 + 1820*D*dM)
  term3 = (1005 + 1820*(Wad+dWad))*(D*(Cpd + (M + dM)*Cpw))/(1 + 1820*D*dM)
  term4 = -F*(Cpd + (M + dM)*Cpw)
  Tg_future = (term1 + term2)/(term3 + term4)
end

% from ocallagan
function [Ta_next] = ta(rhod, HcAs, dt, Cpd, M, dM, Cpw, Tg_f, Tg, Lv, Ta)
  D = 2*rhod/(HcAs*dt)
  Ta_next = (Tg_f*D*(Cpw + (M + dM)*Cpw) - D*Tg*(Cpd + Cpw*M) - D*dM*Lv - Ta)/(1 + 1820*D*dM)
end

% drying rate eqn
% from 1506 of Lopez et al.
function [M_future] = m(M_current, k, dt, Me)
M_future = M_current*exp(-k*dt) + Me*(1 - exp(-k*dt));
end

% relative humidity
function [RH] = rh(Pw, Ps)
  RH = 100*Pw/Ps;
end

% partial pressure water
function [Pw] = pw(Wa, P)
% compute water mole-fraction
  Ma = Wa/(1+Wa);
  MW_water = 18;
  MW_air = 28.96;
  Na = Ma*(MW_water/(Ma*MW_water + (1 - Ma)*MW_air));
  Pw = Na*P;
end

% water activity
function[Aw] = aw(Pw, Pw0)
  Aw = P/Pw0
  % for now, return constant
  Aw = 0.5
end
