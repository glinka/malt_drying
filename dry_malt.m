% % change in height per iteration, m
% dz = 0.01
% % final height, m
% z = 0.3
% % change in time per iteration, s
% dt = 500
% % final time, s
% tfinal = 60*60*30
% dry_malt(dt, tfinal, dz, z, lopez_program)
function dry_malt(dt, tfinal, dz, z, air_program)
  % barley density, kg/m^3
  rho_barley = 600.0;
  % dry barley heat capacity, kJ/(kg K)
  Cp_barley = 1.300;
  % specific heat of water vapor kJ/(kg K)
  Cp_watervapor = 1.870;
  % specific heat of water  kJ/(kg K)
  Cp_water = 4.180;
  % air inflow rate, kg/(m^2 s)
  G = 310/3600.0;
  
  % number of numerical slices, m
  nzs = z/dz;
  % number of timesteps, s
  nts = tfinal/dt;
  % final time, s
  tfinal = nts*dt;
  
  % set up arrays to save:
  % malt temperatures, C
  Tms = zeros(nzs, nts+1);
  % malt moisture contents, kg water/kg dry matter
  Ms = zeros(nzs, nts+1);
  % air temperatures, C
  Tas = zeros(nzs+1, nts);
  % air moisture content, kg water/kg dry air
  Was = zeros(nzs+1, nts);

  % initial malt temperature, C
  Tm_init = 20;
  % setting intitial temperature throughout the bed
  Tms(:, 1) = Tm_init*ones(nzs,1);
  
  % initial malt moisture content, kg water/kg dry matter
  Ms_init = 0.8;
  % setting intitial malt moisture content throughout the bed
  Ms(:, 1) = Ms_init*ones(nzs, 1);

  times = linspace(0, tfinal, nts+1);

  % numerical tolerance for the bisection method that is performed when simulating rewetting, and the maximum allowable iterations to achieve this precision
  BISECTION_TOL = 1e-5;
  BISECTION_MAXITER = 10000;

  % define smaller functions

  % saturated water vapor pressure, Pa
  Ps = @(Ta) 100000*exp(14.293  - 5291/(Ta + 273.15))/(3.2917 - 0.01527*(Ta + 273.15) + 2.54e-5*power(Ta + 273.15, 2));
  % relative humidity
  RH = @(Pw,Ps) Pw/Ps;
  % drying rate paramater, 1/s
  k = @(Ta) 139.3*exp(-4426/(Ta+273));
  % equilibrium moisture content of barley, kg water/kg dry matter
  M_eq = @(Ta,Wa) power(Wa, 0.5)*7040/power(1.8*Ta+32, 2) + 0.06015;
  % change in moisture content of in slice after passage of "dt" time
  dM = @(Ta,M,Wa,dt) -k(Ta)*(M-M_eq(Ta, Wa))*dt/(1+k(Ta)*dt/2);
  % change in air moisture content over slice
  dWa = @(rho,dz,dM,G,dt) -rho*dz*dM/(G*dt);
  % heat transfer coefficient, J/(K s m^3)
  h_barley = @(G,Ta) 856800*power(G*(Ta+273)/101325, 0.6);

  % begin outer loop, iterating through every timestep
  for i = 1:nts
    % calculate current time
    t = i*dt;
    % get inlet air conditions based on current time and specified air program
    [Ta_in, Wa_in] = air_program(t);
    Tas(1, i) = Ta_in;
    Was(1, i) = Wa_in;
    % begin inner loop, iterating over every slice of the bed from bottom to top
    for j = 1:nzs
      % calculate changes in malt and air moisture and malt and grain temperature
      dm = dM(Tas(j, i), Ms(j, i), Was(j, i), dt);
      dwa = dWa(rho_barley, dz, dm, G, dt);
      dtm = dTm(Was(j, i), dwa, dm, Tas(j, i), Cp_barley, Ms(j, i), Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas(j, i)), Tms(j, i), rho_barley) ;
      dta = dTa(Tms(j, i), dtm, rho_barley, h_barley(G, Tas(j, i)), dt, dm, Cp_barley, Cp_water, Ms(j, i), Cp_watervapor, Tas(j, i), G, Was(j,i), dz);
      % enter rewetting routine if the relative humidity is above 98%, as described in Bala's thesis
      % note that, while the thesis 
      if RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) > .98
        %%%% run condensation procedure as described in
        %%%% Bala's thesis, pg. 95
	% record the value of 'dm' entering the routine for use later, and set a somewhat arbitrary 'change in the change of malt moisture content': ddm
	% we use this 'ddm' value to decrement the initial 'dm' value until we find a new value of 'dm' such that the relative humidity is less than 98%
	% then we will have bracketed the desired value of 'dm', the one at which the relative humidity is exactly 98%, between the 'dm_init' and 'dm', after which we can perform the bisection method to find it
        dm_init = dm;
        ddm = dm_init/10.0;
        iters = 0;
        while RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) > .98 && iters < BISECTION_MAXITER
          dm = dm - ddm;
          dwa = dWa(rho_barley, dz, dm, G, dt);
          dtm = dTm(Was(j, i), dwa, dm, Tas(j, i), Cp_barley, Ms(j, i), Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas(j, i)), Tms(j, i), rho_barley);
          dta = dTa(Tms(j, i), dtm, rho_barley, h_barley(G, Tas(j, i)), dt, dm, Cp_barley, Cp_water, Ms(j, i), Cp_watervapor, Tas(j, i), G, Was(j,i), dz);
          iters = iters + 1;
	end
        if iters == BISECTION_MAXITER
          error('search iterations exceeded, desired relative humidity not bracketed')
	end
	% set the 'left' and 'right' 'dm' values to begin bisection
        dm_l = dm;
        dm_r = dm_init;
        iters = 0;
	% perform a bisection-based search for the conditions at which relative humidity is 98% (up to BISECTION_TOL error)
        while abs(RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) - .98) > BISECTION_TOL && iters < BISECTION_MAXITER
          if RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) - .98 > 0
            dm_r = dm;
            dm = dm - (dm_r - dm_l)/2;
          else
            dm_l = dm;
            dm = dm + (dm_r - dm_l)/2;
	  end
          dwa = dWa(rho_barley, dz, dm, G, dt);
          dtm = dTm(Was(j, i), dwa, dm, Tas(j, i), Cp_barley, Ms(j, i), Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas(j, i)), Tms(j, i), rho_barley);
          dta = dTa(Tms(j, i), dtm, rho_barley, h_barley(G, Tas(j, i)), dt, dm, Cp_barley, Cp_water, Ms(j, i), Cp_watervapor, Tas(j, i), G, Was(j,i), dz);
          iters = iters + 1;
	end
        if iters == BISECTION_MAXITER
          error('bisection iterations exceeded, could not converge to desired tolerance')
	end
      end
      % save calculated conditions and move on to calculate bed conditions at the next time
      Ms(j, i+1) = Ms(j, i) + dm;
      Tms(j, i+1) = Tms(j, i) + dtm;
      Was(j+1, i) = Was(j, i) + dwa;
      Tas(j+1, i) = Tas(j, i) + dta;
    end
  end

  % calculate enzyme activities and plot results of simulation

  % show what the avg moisture content looks like
  fig = figure(1);
  ax = axes();
  plot(ax, times/3600.0, mean(Ms, 1))
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  xlabel(ax, 'Time (h)')
  ylabel(ax, 'Average malt moisture content')
  figure(1)
  
  % beta glucanase activity
  bg_prof = beta_gluc_profile(Tas(1,:)+273, mean(Ms, 1), times);
  fig = figure(2);
  ax = axes();
  plot(ax, times(1:length(times)-1)/3600.0, bg_prof)
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  xlabel(ax, 'Time (h)')
  ylabel(ax, '$\beta$ - Glucanase activity (BGU)', 'Interpreter', 'latex')
  figure(2)

  % alpha amylase activity
  am_prof = alpha_am_profile(Tas(1,:)+273, times);
  fig = figure(3);
  ax = axes();
  plot(ax, times(1:length(times)-1)/3600.0, am_prof)
  xlabel(ax, 'Time (h)')
  ylabel(ax, '$\alpha$ - Amylase activity (DU/g dm)', 'Interpreter', 'latex')
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  current_ylim = ylim;
  ylim(ax, [20, current_ylim(2)])
  figure(3)

  % diastatic power
  ds_prof = dias_pow_profile(Tas(1,:)+273, mean(Ms, 1), times);
  fig = figure(4);
  ax = axes();
  plot(ax, times(1:length(times)-1)/3600.0, ds_prof)
  xlabel(ax, 'Time (h)')
  ylabel(ax, ('Diastatic power (WK/ 100 g dm)'))
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  figure(4)

  % limit dextrinase
  ld_prof = limit_dextrinase_profile(Tas(1,:)+273, mean(Ms, 1), times);
  fig = figure(5);
  ax = axes();
  plot(ax, times(1:length(times)-1)/3600.0, ld_prof)
  xlabel(ax, 'Time (h)')
  ylabel(ax, ('Limit-dextrinase (RPU/ 100 g dm)'))
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  figure(5)

  % malt temperature evolution at various depths
  fig = figure(6);
  ax = axes();
  hold(ax);
  n_zslices = 5;
  c = ['r'; 'b'; 'k'; 'g'; 'c'; 'm'; 'y'];
  zspacing = nzs/n_zslices;
  for i = 1:n_zslices
    plot(ax, (1:(nts+1))*dt/3600.0, Tms(zspacing*(i-1)+1, :), 'DisplayName', ['depth: ', num2str(zspacing*(i-1)*dz), ' m'], 'Color', c(i));
  end
  plot(ax, (1:(nts+1))*dt/3600.0, Tms(zspacing*n_zslices, :), 'DisplayName', ['depth: ', num2str(zspacing*(n_zslices)*dz), ' m'], 'Color', c(n_zslices+1));
  xlabel(ax, 'Time (h)');
  ylabel(ax, 'Malt temperature ($\circ$C)', 'Interpreter', 'latex');
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)]);
  legend(ax, 'show');
  figure(6);

  fig = figure(7)
  ax = fig.add_subplot(111)
  ax.hold(True)
  n_zslices = 5
  for i = 1:n_zslices
    ax.plot(np.arange(nts)*dt/3600.0, Tas(zspacing*i, :), label='depth: ' + num2str(zspacing*i*dz) + ' m', c=c(i), lw=1)
  end
  ax.plot(np.arange(nts)*dt/3600.0, Tas(-1, :), label='depth: ' + num2str(zspacing*n_zslices*dz) + ' m', c=c(n_zslices), lw=1)
  ax.set_xlabel('Time (h)')
  ax.set_ylabel('Air temperature (' + r'$\degree$' + 'C)')
  ax.set_xlim(left=0)
  ax.legend(loc=4)
  plt.show()

  % fig = figure(8)
  % ax = fig.add_subplot(111)
  % ax.hold(True)
  % n_zslices = 5
  % for i = 1:n_zslices
  %   ax.plot(np.arange(nts)*dt/3600.0, Was(zspacing*i, :), label='depth: ' + num2str(zspacing*i*dz) + ' m', c=c(i), lw=1)
  % end
  % ax.plot(np.arange(nts)*dt/3600.0, Was(-1, :), label='depth: ' + num2str(zspacing*n_zslices*dz) + ' m', c=c(n_zslices), lw=1)
  % ax.set_xlabel('Time (h)')
  % ax.set_ylabel('Air moisture content (kg water/kg dry air)')
  % ax.set_xlim(left=0)
  % ax.legend(loc=1)
  % plt.show()

  % fig = figure(9)
  % ax = fig.add_subplot(111)
  % ax.hold(True)
  % n_zslices = 5
  % for i = 1:n_zslices
  %   ax.plot(np.arange(nts+1)*dt/3600.0, Ms(zspacing*i, :), label='depth: ' + num2str(zspacing*i*dz) + ' m', c=c(i), lw=1)
  % end
  % ax.plot(np.arange(nts+1)*dt/3600.0, Ms(-1, :), label='depth: ' + num2str(zspacing*n_zslices*dz) + ' m', c=c(n_zslices), lw=1)
  % ax.set_xlabel('Time (h)')
  % ax.set_ylabel('Malt moisture content (kg water/kg dry air)')
  % ax.set_xlim(left=0)
  % ax.legend(loc=1)
  % plt.show()
end


function [dtg] = dTm(Wa, dWa, dM, Ta, Cp_malt, M, Cp_water, Cp_watervapor, dz, rho, G, dt, h, Tm, rho_malt)
  % enthalpy of vaporization of water (kJ/ kg) (assumed constant, taken at 25 C)
  L_water = 1000*43.99/18;
  % !!!!!!!!!!!!!!!!!!!!
  %    UNITS of hcv??
  % !!!!!!!!!!!!!!!!!!!!
  % either J/s m^3 K or kJ/min m^3 K
  % hcv = 49.32*1000*np.power(G, 0.6906)
  hcv = 49.32*power(G, 0.6906)*60;
  % specific heat of malt (kJ/kg K) from thesis
  L_malt = L_water*(1 + 0.5904*exp(-0.1367*M));
  % Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta
  Cp_malt = 1600;
  F = Cp_watervapor*Ta + L_water - Cp_water*Tm;
  A = 2*(Ta - Tm);
  B = Cp_malt + Cp_water*M;
  % latent heat of malt (J/kg) from thesis
  % Lg = 1000*Lv*(1 + 0.5904*exp(-0.1367*M))
  Y = L_malt + Cp_watervapor*Ta - Cp_water*Tm;
  Cp_air = 1006;
  E = Cp_air + Cp_watervapor*(Wa - rho_malt*dz*dM/(G*dt));
  F = Cp_watervapor*Ta + L_water - Cp_water*Tm;
  num = A + rho_malt*dM*(2*Y/hcv + dz*F/(G*E))/dt;
  denom = 1 + rho_malt*(2*B/hcv + dz*(B + Cp_water*dM)/(G*E))/dt;
  dtg = num/denom;
end

function [dta] = dTa(Tm, dtm, rho_malt, h, dt, dM, Cp_malt, Cp_water, M, Cp_watervapor, Ta, G, Wa, dz)
  % enthalpy of vaporization of water (kJ/ kg) (assumed constant, taken at 25 C)
  L_water = 1000*43.99/18;
  % !!!!!!!!!!!!!!!!!!!!
  %    UNITS of hcv??
  % !!!!!!!!!!!!!!!!!!!!
  % either J/s m^3 K or kJ/min m^3 K
  % hcv = 49.32*1000*power(G, 0.6906)
  hcv = 49.32*power(G, 0.6906)*60;
  % specific heat of malt (kJ/kg K) from thesis
  L_malt = L_water*(1 + 0.5904*exp(-0.1367*M));
  % Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta
  Cp_malt = 1600;
  F = Cp_watervapor*Ta + L_water - Cp_water*Tm;
  A = 2*(Ta - Tm);
  B = Cp_malt + Cp_water*M;
  % latent heat of malt (J/kg) from thesis
  % Lg = 1000*Lv*(1 + 0.5904*exp(-0.1367*M))
  Y = L_malt + Cp_watervapor*Ta - Cp_water*Tm;
  Cp_air = 1006;
  E = Cp_air + Cp_watervapor*(Wa - rho_malt*dz*dM/(G*dt));
  F = Cp_watervapor*Ta + L_water - Cp_water*Tm;
  ta = -rho_malt*dz*(dtm*(B + Cp_water*dM) - dM*F)/(G*E*dt);
end

% partial pressure water, Pa
function [pw] = Pw(Wa)
  MW_water = 18.0;
  MW_air = 28.96;
  P = 101325;
  Na_dry = Wa*MW_air/MW_water;
  pw = P*Na_dry/(1.0 + Na_dry);
end

% returns beta_glucanase activity profile based
% on the drying outputs: air temp, air moisture
% and their corresponding times
function [bg_prof] = beta_gluc_profile(Tas, Ms, times)
  % uses RK4 to integrate
  beta_init=700;
  nsteps = size(times, 2);
  beta_glucs = zeros(nsteps-1, 1);
  beta_glucs(1) = beta_init;
  % universal gas constant, J/mol K
  R = 8.3143;
  % model parameter, pg 161, 1/min
  k_beta0 = 3.089e8;
  % model parameter, pg 161, J/mol
  E_beta = 7.248e4;
  k_beta = @(T) k_beta0*exp(-E_beta/(R*T));
  % beta_prime, 1/s
  f = @(T,Wa,beta) -k_beta(T)*Wa*beta/60.0;
  % because there is no initial value assigned to
  % the air temperature or moisture in the bed,
  % there will be one less Ta and Wa than there are
  % times, so the indexing will differ by one
  for i = 1:(nsteps-2)
    dt = times(i+1) - times(i);
    k1 = f(Tas(i), Ms(i), beta_glucs(i));
    k2 = f((Tas(i) + Tas(i+1))/2.0, (Ms(i) + Ms(i+1))/2.0, beta_glucs(i) + dt*k1/2.0);
    k3 = f((Tas(i) + Tas(i+1))/2.0, (Ms(i) + Ms(i+1))/2.0, beta_glucs(i) + dt*k2/2.0);
    k4 = f(Tas(i+1), Ms(i+1), beta_glucs(i) + dt*k3);
    beta_glucs(i+1) = beta_glucs(i) + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0;
  end
  bg_prof = beta_glucs;
end

% returns alpha_amylase activity profile based
% on the drying outputs: air temp, air moisture
% and their corresponding times
function [aa_prof] = alpha_am_profile(Tas, times)
  % uses RK4 for integration
  alpha_init=55;
  nsteps = size(times, 2);
  alpha_ams = zeros(nsteps-1, 1);
  alpha_ams(1) = alpha_init;
  % universal gas constant, J/mol K
  R = 8.3143;
  % model parameter, pg 161, 1/min
  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  % !!!!!!!!!!!!!!!!!!!!
  %    UNITS of k_alpha0??
  % PAPER'S VALUE IS 5.7654e9
  % BUT THIS APPEARS TO BE ORDERS OF
  % MAGNITUDE OFF. THE CURRENT VALUE
  % of 5.7654e7 PRODUCES RESULTS
  % THAT CLOSELY MATCH THE PAPER'S
  % !!!!!!!!!!!!!!!!!!!!
  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  k_alpha0 = 5.7654e7;
  % model parameter, pg 161, J/mol
  E_alpha = 7.8913e4;
  k_alpha = @(T) k_alpha0*exp(-E_alpha/(R*T));
  % alpha_prime, 1/s
  f = @(T,alpha) -k_alpha(T)*alpha/60.0;
  % because there is no initial value assigned to
  % the air temperature or moisture in the bed,
  % there will be one less Ta and Wa than there are
  % times, so the indexing will differ by one
  for i = 1:(nsteps-2)
    dt = times(i+1) - times(i);
    k1 = f(Tas(i), alpha_ams(i));
    k2 = f((Tas(i) + Tas(i+1))/2.0, alpha_ams(i) + dt*k1/2.0);
    k3 = f((Tas(i) + Tas(i+1))/2.0, alpha_ams(i) + dt*k2/2.0);
    k4 = f(Tas(i+1), alpha_ams(i) + dt*k3);
    alpha_ams(i+1) = alpha_ams(i) + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0;
  end
  aa_prof = alpha_ams;
end

% returns alpha_amylase activity profile based
% on the drying outputs: air temp, air moisture
% and their corresponding times
function [dp_f] = dias_pow_f(T, M, dias)
  % dias_prime, 1/s
  % model parameter, pg 161, 1/min
  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  % !!!!!!!!!!!!!!!!!!!!
  %    UNITS of k_dias0??
  % !!!!!!!!!!!!!!!!!!!!
  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  k_dias0 = 4.8037e10; %e12
  % model parameter, pg 161, J/mol
  E_dias = 9.5056e4;
  % model parameter, pg 161, units?
  r_dias = 5.0361e-2;
  % model parameter, pg 161, WK/ 100 g dm
  K_dias = 470.0;
  % model parameter, pg 161, % dry basis
  M_dias = .40;
  % universal gas constant, J/mol K
  R = 8.3143;
  k_dias = @(T) k_dias0*exp(-E_dias/(R*T));
  % s, ?units?, must be 1/min
  s = @(dias,T,M) r_dias*T*(M - M_dias)*(1-dias/K_dias);
  % only when M >= M_dias is s nonzero
  if M >= M_dias
    dp_f = (s(dias, T, M) - k_dias(T)*dias)/60.0;
  else
    dp_f = -k_dias(T)*dias/60.0;
  end
end

function [dp_prof] = dias_pow_profile(Tas, Ms, times)
  % returns diastatic power profile based
  % on the drying outputs: air temp, air moisture
  % and their corresponding times
  % !!!!!!!!!!!!!!!!!!!!
  % uses RK4 for integration
  dias_init=400;
  nsteps = size(times, 2);
  dias_pows = zeros(nsteps-1, 1);
  dias_pows(1) = dias_init;
  % because there is no initial value assigned to
  % the air temperature or moisture in the bed,
  % there will be one less Ta and Wa than there are
  % times, so the indexing will differ by one
  for i = 1:(nsteps-2)
    dt = times(i+1) - times(i);
    k1 = dias_pow_f(Tas(i), Ms(i), dias_pows(i));
    k2 = dias_pow_f((Tas(i) + Tas(i+1))/2.0, (Ms(i) + Ms(i+1))/2.0, dias_pows(i) + dt*k1/2.0);
    k3 = dias_pow_f((Tas(i) + Tas(i+1))/2.0, (Ms(i) + Ms(i+1))/2.0, dias_pows(i) + dt*k2/2.0);
    k4 = dias_pow_f(Tas(i+1), Ms(i+1), dias_pows(i) + dt*k3);
    dias_pows(i+1) = dias_pows(i) + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0;
  end
  dp_prof = dias_pows;
end

function [ld_f] = lim_dext_f(T, M, dext)
  % dext_prime, 1/s
  % model parameter, pg 161, 1/min
  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  % !!!!!!!!!!!!!!!!!!!!
  %    UNITS of k_dext0??
  % !!!!!!!!!!!!!!!!!!!!
  % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  k_dext0 = 1.6554e18; %e20
  % model parameter, pg 161, J/mol
  E_dext = 1.4516e5;
  % model parameter, pg 161, units?
  r_dext = 6.4693;
  % model parameter, pg 161, RPU/ 100 g dm
  K_dext = 93.963;
  % model parameter, pg 161, % dry basis
  M_dext = 0.40;
  % universal gas constant, J/mol K
  R = 8.3143;
  k_dext = @(T) k_dext0*exp(-E_dext/(R*T));
  % s, ?units?, must be 1/min
  s = @(dext) r_dext*(1-dext/K_dext);
  % only when M >= M_dext is s nonzero
  if M >= M_dext
    ld_f = (s(dext) - k_dext(T)*dext)/60.0;
  else
    ld_f = -k_dext(T)*dext/60.0;
  end
end

function [ld_prof] = limit_dextrinase_profile(Tas, Ms, times)
  % returns limit-dextrinase profile based
  % on the drying outputs: air temp, air moisture
  % and their corresponding times
  % !!!!!!!!!!!!!!!!!!!!
  % uses RK4 for integration
  dex_init=80;
  nsteps = size(times, 2);
  lim_dexts = zeros(nsteps-1, 1);
  lim_dexts(1) = dex_init;
  % because there is no initial value assigned to
  % the air temperature or moisture in the bed,
  % there will be one less Ta and Wa than there are
  % times, so the indexing will differ by one
  for i = 1:(nsteps-2)
    dt = times(i+1) - times(i);
    k1 = lim_dext_f(Tas(i), Ms(i), lim_dexts(i));
    k2 = lim_dext_f((Tas(i) + Tas(i+1))/2.0, (Ms(i) + Ms(i+1))/2.0, lim_dexts(i) + dt*k1/2.0);
    k3 = lim_dext_f((Tas(i) + Tas(i+1))/2.0, (Ms(i) + Ms(i+1))/2.0, lim_dexts(i) + dt*k2/2.0);
    k4 = lim_dext_f(Tas(i+1), Ms(i+1), lim_dexts(i) + dt*k3);
    lim_dexts(i+1) = lim_dexts(i) + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0;
  end
  ld_prof = lim_dexts;
end
