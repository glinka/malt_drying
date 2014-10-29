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
  G = 310/3600.0;
  
  nzs = z/dz;
  nts = tfinal/dt;
  tfinal = nts*dt;

  Tgs = zeros(nzs, nts+1);
  Ms = zeros(nzs, nts+1);
  Tas = zeros(nzs+1, nts);
  Was = zeros(nzs+1, nts);

  Tg_init = 20;
  Tgs(:, 1) = Tg_init*ones(nzs,1);
  
  Ms_init = 0.8;
  Ms(:, 1) = Ms_init*ones(nzs, 1);

  times = linspace(0, tfinal, nts+1);

  CONDENSATION_TOL = 1e-5;
  maxiter = 10000;

  % define smaller functions

  % saturated water vapor pressure, Pa
  Ps = @(Ta) 100000*exp(14.293  - 5291/(Ta + 273.15))/(3.2917 - 0.01527*(Ta + 273.15) + 2.54e-5*power(Ta + 273.15, 2));
  % relative humidity
  RH = @(Pw,Ps) 100.0*Pw/Ps;
  % drying rate paramater, 1/s
  k = @(Ta) 139.3*exp(-4426/(Ta+273));
  % equilibrium moisture content of barley
  M_eq = @(Ta,Wa) power(Wa, 0.5)*7040/power(1.8*Ta+32, 2) + 0.06015;
  % change in moisture content of in slice
  % during time change "dt"
  dM = @(Ta,M,Wa,dt) -k(Ta)*(M-M_eq(Ta, Wa))*dt/(1+k(Ta)*dt/2);
  % change in air moisture content over slice
  dWa = @(rho,dz,dM,G,dt) -rho*dz*dM/(G*dt);
  % heat transfer coefficient, J/(K s m^3)
  % G - air flow rate, kg/(m^2 s)
  % Ta - air temp, degC
  h_barley = @(G,Ta) 856800*power(G*(Ta+273)/101325, 0.6);
  for i = 1:nts
    t = i*dt;
    [Ta_in, Wa_in] = air_program(t);
    Tas(1, i) = Ta_in;
    Was(1, i) = Wa_in;
    for j = 1:nzs
      dm = 0;
      dwa = 0;
      dm = dM(Tas(j, i), Ms(j, i), Was(j, i), dt);
      dwa = dWa(rho_barley, dz, dm, G, dt);
      dtg = deltaTg(Was(j, i), dwa, dm, Tas(j, i), Cp_barley, Ms(j, i), Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas(j, i)), Tgs(j, i), rho_barley) ;
      dta = deltaTa(Tgs(j, i), dtg, rho_barley, h_barley(G, Tas(j, i)), dt, dm, Cp_barley, Cp_water, Ms(j, i), Cp_watervapor, Tas(j, i), G, Was(j,i), dz);
      if RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) > 98
        dm_s = dm;
        ddm = dm_s/10.0;
        % run condensation procedure as described in
        % Bala's thesis, pg. 95
        iters = 0;
        while RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) > 98 && iters < maxiter
          dm = dm - ddm;
          dwa = dWa(rho_barley, dz, dm, G, dt);
          dtg = deltaTg(Was(j, i), dwa, dm, Tas(j, i), Cp_barley, Ms(j, i), Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas(j, i)), Tgs(j, i), rho_barley);
          dta = deltaTa(Tgs(j, i), dtg, rho_barley, h_barley(G, Tas(j, i)), dt, dm, Cp_barley, Cp_water, Ms(j, i), Cp_watervapor, Tas(j, i), G, Was(j,i), dz);
          iters = iters + 1;
	end
        if iters == maxiter
          error('search iterations exceeded')
	end
        dm_l = dm;
        dm_r = dm_s;
        iters = 0;
        while abs(RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) - 98) > CONDENSATION_TOL && iters < maxiter
          if RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) - 98 > 0
            dm_r = dm;
            dm = dm - (dm_r - dm_l)/2;
          else
            dm_l = dm;
            dm = dm + (dm_r - dm_l)/2;
	  end
          dwa = dWa(rho_barley, dz, dm, G, dt);
          dtg = deltaTg(Was(j, i), dwa, dm, Tas(j, i), Cp_barley, Ms(j, i), Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas(j, i)), Tgs(j, i), rho_barley);
          dta = deltaTa(Tgs(j, i), dtg, rho_barley, h_barley(G, Tas(j, i)), dt, dm, Cp_barley, Cp_water, Ms(j, i), Cp_watervapor, Tas(j, i), G, Was(j,i), dz);
          iters = iters + 1;
	end
        if iters == maxiter
          error('bisection iterations exceeded')
	end
      end
      Ms(j, i+1) = Ms(j, i) + dm;
      Tgs(j, i+1) = Tgs(j, i) + dtg;
      Was(j+1, i) = Was(j, i) + dwa;
      Tas(j+1, i) = Tas(j, i) + dta;
    end
  end

  % show what the avg moisture content looks like
  fig = figure(1);
  ax = axes();
  plot(ax, times/3600.0, mean(Ms, 1))
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  xlabel(ax, 'Time (h)')
  ylabel(ax, 'Average malt moisture content')
  figure(1)
  
  bg_prof = beta_gluc_profile(Tas(1,:)+273, mean(Ms, 1), times);
  
  fig = figure(2);
  ax = axes();
  plot(ax, times(1:length(times)-1)/3600.0, bg_prof)
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  xlabel(ax, 'Time (h)')
  ylabel(ax, '$\beta$ - Glucanase activity (BGU)', 'Interpreter', 'latex')
  figure(2)

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

  ds_prof = dias_pow_profile(Tas(1,:)+273, mean(Ms, 1), times);
  
  fig = figure(4);
  ax = axes();
  plot(ax, times(1:length(times)-1)/3600.0, ds_prof)
  xlabel(ax, 'Time (h)')
  ylabel(ax, ('Diastatic power (WK/ 100 g dm)'))
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  figure(4)

  ld_prof = limit_dextrinase_profile(Tas(1,:)+273, mean(Ms, 1), times);
  
  fig = figure(5);
  ax = axes();
  plot(ax, times(1:length(times)-1)/3600.0, ld_prof)
  xlabel(ax, 'Time (h)')
  ylabel(ax, ('Limit-dextrinase (RPU/ 100 g dm)'))
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)])
  figure(5)

  fig = figure(6);
  ax = axes();
  hold(ax);
  n_zslices = 5;
  c = ['r'; 'b'; 'k'; 'g'; 'c'; 'm'; 'y'];
  zspacing = nzs/n_zslices;
  for i = 1:n_zslices
    plot(ax, (1:(nts+1))*dt/3600.0, Tgs(zspacing*(i-1)+1, :), 'DisplayName', ['depth: ', num2str(zspacing*(i-1)*dz), ' m'], 'Color', c(i));
  end
  plot(ax, (1:(nts+1))*dt/3600.0, Tgs(zspacing*n_zslices, :), 'DisplayName', ['depth: ', num2str(zspacing*(n_zslices)*dz), ' m'], 'Color', c(n_zslices+1));
  xlabel(ax, 'Time (h)');
  ylabel(ax, 'Grain temperature ($\circ$C)', 'Interpreter', 'latex');
  current_xlim = xlim;
  xlim(ax, [0, current_xlim(2)]);
  legend(ax, 'show');
  figure(5);

  % fig = figure(7)
  % ax = fig.add_subplot(111)
  % ax.hold(True)
  % n_zslices = 5
  % for i = 1:n_zslices
  %   ax.plot(np.arange(nts)*dt/3600.0, Tas(zspacing*i, :), label='depth: ' + num2str(zspacing*i*dz) + ' m', c=c(i), lw=1)
  % end
  % ax.plot(np.arange(nts)*dt/3600.0, Tas(-1, :), label='depth: ' + num2str(zspacing*n_zslices*dz) + ' m', c=c(n_zslices), lw=1)
  % ax.set_xlabel('Time (h)')
  % ax.set_ylabel('Air temperature (' + r'$\degree$' + 'C)')
  % ax.set_xlim(left=0)
  % ax.legend(loc=4)
  % plt.show()

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
  % ax.set_ylabel('Grain moisture content (kg water/kg dry air)')
  % ax.set_xlim(left=0)
  % ax.legend(loc=1)
  % plt.show()
end


% change in temperature of malt in slice
function [tg_next] = Tg_next(Wa, dWa, dM, Ta, Cp_grain, M, Cp_water, Cp_watervapor, dz, rho, G, dt, h, Tg)
  F = -dz*rho/(G*dt);
  numerator_i = Ta*(1005 + 1820*Wa) - Tg*F*(Cp_grain + M*Cp_water) - 2501000*dWa;
  D = 2*rho/(h*dt);
  Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta;
  numerator_ii = (1005+1820*(Wa+dWa))*(Ta + D*dM*Lv + D*Tg*(Cp_grain + M*Cp_water))/(1 + 1820*D*dM);
  denominator = (1005+1820*(Wa+dWa))*(D*(Cp_grain + (M+dM)*Cp_water) + 1)/(1 + 1820*D*dM) - F*(Cp_grain + (M+dM)*Cp_water);
  % print "D:", D, "Lv:", Lv, "Cp_grain:", Cp_grain, "Cp_water:", Cp_water, "because stdout is the best debugger:", (Ta + D*dM*Lv + D*Tg*(Cp_grain + M*Cp_water))
  % print "num_i:", numerator_i, "num_ii:", numerator_ii, "denom:", denominator
  tg_next = (numerator_i + numerator_ii)/denominator;
end

% change in air temperature across slice
function [ta_next] = Ta_next(Tg, dTg, rho, h, dt, dM, Cp_grain, Cp_water, M, Cp_watervapor, Ta)
  D = 2*rho/(h*dt);
  Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta;
  ta_next = ((Tg+dTg)*(D*(Cp_grain + (M+dM)*Cp_water) + 1) - D*Tg*(Cp_grain + M*Cp_water) - D*dM*Lv - Ta)/(1 + 1820*D*dM);
end

function [delta_tg] = deltaTg(Wa, dWa, dM, Ta, Cp_grain, M, Cp_water, Cp_watervapor, dz, rho, G, dt, h, Tg, rho_grain)
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
  Cp_grain = 1600;
  F = Cp_watervapor*Ta + L_water - Cp_water*Tg;
  A = 2*(Ta - Tg);
  B = Cp_grain + Cp_water*M;
  % latent heat of grain (J/kg) from thesis
  % Lg = 1000*Lv*(1 + 0.5904*exp(-0.1367*M))
  Y = L_malt + Cp_watervapor*Ta - Cp_water*Tg;
  Cp_air = 1006;
  E = Cp_air + Cp_watervapor*(Wa - rho_grain*dz*dM/(G*dt));
  F = Cp_watervapor*Ta + L_water - Cp_water*Tg;
  num = A + rho_grain*dM*(2*Y/hcv + dz*F/(G*E))/dt;
  denom = 1 + rho_grain*(2*B/hcv + dz*(B + Cp_water*dM)/(G*E))/dt;
  delta_tg = num/denom;
end

function [delta_ta] = deltaTa(Tg, dTg, rho_grain, h, dt, dM, Cp_grain, Cp_water, M, Cp_watervapor, Ta, G, Wa, dz)
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
  Cp_grain = 1600;
  F = Cp_watervapor*Ta + L_water - Cp_water*Tg;
  A = 2*(Ta - Tg);
  B = Cp_grain + Cp_water*M;
  % latent heat of grain (J/kg) from thesis
  % Lg = 1000*Lv*(1 + 0.5904*exp(-0.1367*M))
  Y = L_malt + Cp_watervapor*Ta - Cp_water*Tg;
  Cp_air = 1006;
  E = Cp_air + Cp_watervapor*(Wa - rho_grain*dz*dM/(G*dt));
  F = Cp_watervapor*Ta + L_water - Cp_water*Tg;
  delta_ta = -rho_grain*dz*(dTg*(B + Cp_water*dM) - dM*F)/(G*E*dt);
end

function [ta_next] = Ta_next_check(dz, rho, G, dt, h, Ta, Cp_watervapor, Wa, Tg, dTg, Cp_grain, M, dM, Cp_water, dWa)
  F = -dz*rho/(G*dt);
  ta_next = (Ta*(1005 + 1820*Wa) +(Tg+dTg)*F*(Cp_grain + (M+dM)*Cp_water) - Tg*F*(Cp_grain + M*Cp_water) - 2501000*dWa)/(1005 + 1820*(Wa+dWa));
end

% partial pressure water, Pa
function [pw] = Pw(Wa)
  MW_water = 18.0;
  MW_air = 28.96;
  P = 101325;
  Na_dry = Wa*MW_air/MW_water;
  pw = P*Na_dry/(1.0 + Na_dry);
end

function [bg_prof] = beta_gluc_profile(Tas, Ms, times)
  % returns beta_glucanase activity profile based
  % on the drying outputs: air temp, air moisture
  % and their corresponding times
  % !!!!!!!!!!!!!!!!!!!!
  % uses RK4
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

function [aa_prof] = alpha_am_profile(Tas, times)
  % returns alpha_amylase activity profile based
  % on the drying outputs: air temp, air moisture
  % and their corresponding times
  % !!!!!!!!!!!!!!!!!!!!
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
