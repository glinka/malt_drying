%
% WRITTEN BY ALEXANER HOLIDAY: holiday@alexanderholiday.com
% 

function dry_malt()

  % set up the gui

  centi_hscaling = 0.0282;
  centi_vscaling = 0.03;
  centi_scaling = [centi_hscaling, centi_vscaling, centi_hscaling, centi_vscaling];
  f = figure(101);%, 'Position', [100 100 1000 700]);
  set(f, 'Units', 'centimeters',...
      'Position', [2.8200, 3.0000, 19.4580, 15.0000]);


  % group relevant gui pieces into panels, here we create one for numerical parameters
  params_panel = uipanel(f, 'Units', 'centimeters',...
			 'Position', [7.6, 6.0000, 11.0500, 2.5000],...
  			 'Title', 'Parameter inputs:');

  % create table for parameter inputs 
  % 228/6.43, 40/1.2
			 % [0.2820, 0.1500, 6.4, 1.2300]
  params_table = uitable(params_panel, 'Units', 'centimeters',...
			 'Position', [0.2820, 0.1500, 10.4, 1.65],...
			 'ColumnName', {'z (m)', 'dz (m)', 'dt (s)', 'Initial malt moisture|(kg water/kg dry malt)'},...
			 'ColumnFormat', {'short', 'short', 'short', 'short'},...
			 'Data', [0.3, 0.01, 500, 0.8],...
			 'ColumnEditable', [true, true, true, true],...
			 'RowName', []);
  % params_table.Position(3:4) = params_table.Extent(3:4);

  % panel for specification of air program
  program_panel = uipanel(f, 'Units', 'centimeters',...
			  'Position', [0, 10.5000, 19.4580, 4.5000],...
  			  'Title', 'Air program specification:');

  % create default program
  default_entries = [0, 310/3600.0, 60, 6.5, 0;
		     20, 310/3600.0, 65, 6.5, 20.0;
		     24, 310/3600.0, 80, 6.5, 40.0];

  % create air program table
  air_prog_table = uitable(program_panel, 'Units', 'centimeters',...
			   'Position', [0.5640, 1.5000, 15.1, 2.3],...
			   'Data', default_entries,...
			   'ColumnName', {'Start time (h)', 'Inlet flowrate (kg/(m^2 s))', 'Inlet temp (C)', 'Inlet RH (%)', 'Recirculation (%)'},...
			   'ColumnFormat', {'bank', 'bank', 'bank', 'bank', 'bank'},...
			   'ColumnEditable', [true, true, true, true, true],...
			   'RowName', []);
  % air_prog_table.Position(3:4) = air_prog_table.Extent(3:4);

  stop_time_table = uitable(program_panel, 'Units', 'centimeters',...
			    'Position', [16.2714, 0.3000, 2.3688, 1.2600],...
			    'Data', [25],...
			    'ColumnName', 'End time (h)',...
			    'ColumnFormat', {'short'},...
			    'ColumnEditable', [true],...
			    'RowName', []);
  % stop_time_table.Position(3:4) = stop_time_table.Extent(3:4);

  % set table dimensions
  % table.Position(3) = table.Extent(3);
  % table.Position(4) = table.Extent(4);

  % create button that adds additional row
  add_row_button = uicontrol(program_panel, 'Style', 'pushbutton',...
		      'String', 'Add row',...
		      'Units', 'centimeters',...
		      'Position', [16.0740, 2.8500, 2.8200, 0.6000],...
		      'Callback', {@add_row, air_prog_table});
  % add_row.Position(3) = add_row.Extent(3);

  % create button that removes last row
  delete_row_button = uicontrol(program_panel, 'Style', 'pushbutton',...
		      'String', 'Delete row',...
		      'Units', 'centimeters',...
		      'Position', [16.0740, 1.9500, 2.8200, 0.6000],...
		      'Callback', {@delete_row, air_prog_table});
  % add_row.Position(3) = add_row.Extent(3);

  plot_panel = uipanel(f, 'Units', 'centimeters',...
			  'Position', [0.6, 0.6000, 6.4860, 9.0000],...
  			  'Title', 'Desired plots:');

  % create checkbox labels, then insert evenly spaced checkboxes
  checkbox_labels = {'Avg moisture',...
		     'Malt temp profiles',...
		     'Air temp profiles',...
		     'Air RH profiles',...
		     'Malt moisture profiles',...
		     'Beta glucanase activity',...
		     'Alpha amylase activity',...
		     'Diastatic power',...
		     'Limit dextrinase'};

  labels_dim = size(checkbox_labels);
  nlabels = labels_dim(2);
  plot_opts = [];
  scaling = 0.9;
  for i = 1:nlabels
      plot_opts(i) = uicontrol(plot_panel, 'Style', 'checkbox',...
			       'Units', 'normalized',...
			       'Position', [0.1, scaling*(1-1.0*i/nlabels), 0.75, scaling*0.9/nlabels],...
			       'String', checkbox_labels(i),...
			       'Value', true);
  end

  select_all_button = uicontrol(plot_panel, 'Style', 'pushbutton',...
			    'String', 'Select all',...
			    'Units', 'centimeters',...
			    'Position', [0.2820, 7.8000, 2.8200, 0.6000],...
			    'Callback', {@set_all, 1, plot_opts});

  select_none_button = uicontrol(plot_panel, 'Style', 'pushbutton',...
			    'String', 'Select none',...
			    'Units', 'centimeters',...
			    'Position', [3.3840, 7.8000, 2.8200, 0.6000],...
			    'Callback', {@set_all, 0, plot_opts});


  % create button to run model with specified inputs
  run_button = uicontrol(f, 'Style', 'pushbutton',...
			    'String', 'Run',...
			    'Units', 'centimeters',...
			    'Position', [12, 2.5000, 2.8200, 1.5000],...
			    'Callback', {@parse_and_run, params_table, air_prog_table, stop_time_table, plot_opts});

  function set_all(source, callback, newvalue, plot_opts)
    for p = plot_opts
	set(p, 'Value', newvalue);
    end
  end

  function add_row(source, callbackdata, table)
    % instead of "get(table, 'Data')", should be able to use "table.Data();"

    % reposition table to make new row visible
    table_pos = get(table, 'Position');
    panel_pos = get(program_panel, 'Position');
    offset = 0.5625;
    % check if running off panel
    if table_pos(4)+table_pos(1)+offset > panel_pos(4)
      set(f, 'Position', get(f, 'Position') + [0, 0, 0, offset])
      set(program_panel, 'Position', panel_pos + [0, 0, 0, offset]);
      set(table, 'Position', get(table, 'Position') + [0, 0, 0, offset]);
    else
      set(table, 'Position', get(table, 'Position') + [0, -offset, 0, offset]);
    end
       

    % find current dimensions and resize table to include an additional row
    current_dims = size(get(table, 'Data'));
    new_data = zeros(current_dims + [1, 0]);
    % add back old data
    new_data(1:current_dims(1), 1:current_dims(2)) = get(table, 'Data');
    set(table, 'Data', new_data);
  end

  function delete_row(source, callbackdata, table)
    % instead of "get(table, 'Data')", should be able to use "table.Data();"

    % reposition table to make new row visible
    offset = 0.5625;
    table_pos = get(table, 'Position');
    % do not delete if only one row is present
    if table_pos(4) < 1.2
       msgbox('Must have at least one row of specifications');
    else
      set(table, 'Position', get(table, 'Position') + [0, offset, 0, -offset]);
      % find current dimensions and resize table to include an additional row
      current_dims = size(get(table, 'Data'));
      new_data = zeros(current_dims + [-1, 0]);
      % add back old data
      old_data = get(table, 'Data');
      new_data = old_data(1:current_dims(1)-1, 1:current_dims(2));
      set(table, 'Data', new_data);
    end
  end

  % function that parses state of plot option checkboxes and runs model
  function parse_and_run(source, callbackdata, params_table, air_prog_table, stop_time_table, plot_opts)

    params = get(params_table, 'Data');
    air_prog = get(air_prog_table, 'Data');

    map_keys = {};
    map_vals = [true];
    i = 1;
    for opt = plot_opts
      map_keys(i) = get(opt, 'String');
      map_vals(i) = logical(get(opt, 'Value'));
      i = i + 1;
    end

    params(length(params)+1) = get(stop_time_table, 'Data');
    plot_opts = containers.Map(map_keys, map_vals);
    run_malt_model(params, air_prog, plot_opts);
  end

end


function run_malt_model(params, air_prog, plot_opts)
  % retrieve parameters from gui
  % total height, m
  z = params(1);
  % change in height per iteration, m
  dz = params(2);
  % change in time per iteration, s
  dt = params(3);
  % initial malt moisture content, kg water/kg dry matter
  Ms_init = params(4);
  % final time, s
  tfinal = params(5)*3600;

  % barley density, kg/m^3
  % Bala's thesis, pg. 10
  % assumed constant
  % note that pg. 114 suggested using 608.0 instead
  rho_barley = 527.0;
  % dry barley heat capacity, kJ/(kg K)
  % Bala's thesis, pg. 16
  % assumed constant
  Cp_barley = 1.651;
  % specific heat of water vapor kJ/(kg K)
  % assumed constant, taken at 325 K
  Cp_watervapor = 1.870;
  % specific heat of water  kJ/(kg K)
  % assumed constant
  Cp_water = 4.180;
  % air inflow rate, kg/(m^2 s)
  % Bala's thesis, pg. 114
  G = 23.41/60.0;
  
  % number of numerical slices, m
  nzs = int64(z/dz);
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
  % air relative humidity, fraction
  RHs = zeros(nzs+1, nts);
  % initial malt temperature, C
  Tm_init = 20;
  % setting intitial temperature throughout the bed
  Tms(:, 1) = Tm_init*ones(nzs,1);
  % setting intitial malt moisture content throughout the bed
  Ms(:, 1) = Ms_init*ones(nzs, 1);
  times = linspace(0, tfinal, nts+1);

  % define smaller functions
  % saturated water vapor pressure, Pa
  % Food Plant Design, Antonio Lopez-Gomez, Gustavo V. pg. 88
  Ps = @(Ta) 100000*exp(14.293  - 5291/(Ta + 273.15))/(3.2917 - 0.01527*(Ta + 273.15) + 2.54e-5*power(Ta + 273.15, 2));
  % relative humidity
  % by definition
  RH = @(Pw,Ps) Pw/Ps;
  % drying rate paramater, 1/s
  % Bala's thesis, pg. 77
  k = @(Ta) 9.5294e6*exp(-6725.02/(Ta+273.15));
  % !!!!!!!!!!!!!!!!!!!!
  % to use the value found in the internal Cargill report: "MOISTURE SORPTION ISOTHERMS FOR GERMINATED BARLEY" by Elizabeth Reid, pg. 16
  % one would need to water activity
  % !!!!!!!!!!!!!!!!!!!!
  % equilibrium moisture content of barley, kg water/kg dry matter
  M_eq = @(Ta,Wa) power(Wa, 0.5)*7040/power(1.8*Ta+32, 2) + 0.06015;
  % change in moisture content of in slice after passage of "dt" time
  dM = @(Ta,M,Wa,dt) -k(Ta)*(M-M_eq(Ta, Wa))*dt/(1+k(Ta)*dt/2);
  % change in air moisture content over slice
  % Bala's thesis, pg. 90
  dWa = @(rho,dz,dM,G,dt) -rho*dz*dM/(G*dt);
  % heat transfer coefficient, J/(K s m^3)
  h_barley = @(G,Ta) 856800*power(G*(Ta+273)/101325, 0.6);

  % begin outer loop, iterating through every timestep
  for i = 1:nts
    % calculate current time
    t = i*dt;

    % get inlet air conditions based on current time and specified air program
    [G_in, Ta_in, RH_in, recirc] = parse_air_prog(air_prog, t);
    Wa_in = Wa_from_rh(RH_in, Ps(Ta_in));

    % Tas(1, i) and Was(1, i) will be mix of outlet air exiting top of bed and fresh inlet air at conditions specified by air_prog
    % cannot recirculate during first iteration, no bed-output data
    if i > 1
      Wa_out = Was(nzs+1, i-1);
      Ta_out = Tas(nzs+1, i-1);
      RH_out = RHs(nzs+1, i-1);
      Tas(1, i) = recirc*Ta_out + (1-recirc)*Ta_in;
      Was(1, i) = recirc*Wa_out + (1-recirc)*Wa_in;
      RHs(1, i) = recirc*RH_out + (1-recirc)*RH_in;
    else
      Tas(1, i) = Ta_in;
      Was(1, i) = Wa_in;
      RHs(1, i) = RH_in;
    end

    % begin inner loop, iterating over every slice of the bed from bottom to top
    for j = 1:nzs
      % calculate changes in malt and air moisture and malt and grain temperature
      dm = dM(Tas(j, i), Ms(j, i), Was(j, i), dt);
      dwa = dWa(rho_barley, dz, dm, G, dt);
      dtm = dTm(Was(j, i), dwa, dm, Tas(j, i), Cp_barley, Ms(j, i), Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas(j, i)), Tms(j, i), rho_barley) ;
      dta = dTa(Tms(j, i), dtm, rho_barley, h_barley(G, Tas(j, i)), dt, dm, Cp_barley, Cp_water, Ms(j, i), Cp_watervapor, Tas(j, i), G, Was(j,i), dz);

      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % BEGIN BISECTION PROCEDURE %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % enter rewetting routine if the relative humidity is above 98%, as described in Bala's thesis
      % note that, while the thesis 
      if RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta)) > .98

	% numerical tolerance for the bisection method that is performed when simulating rewetting, and the maximum allowable iterations to achieve this precision
	BISECTION_TOL = 1e-5;
	BISECTION_MAXITER = 10000;

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

	% now that the desired properties are bracketed, perform a bisection-based search for the conditions at which relative humidity is 98% (up to BISECTION_TOL error)
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
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % END BISECTION PROCEDURE %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % save calculated conditions and move on to calculate bed conditions at the next time
      Ms(j, i+1) = Ms(j, i) + dm;
      Tms(j, i+1) = Tms(j, i) + dtm;
      Was(j+1, i) = Was(j, i) + dwa;
      Tas(j+1, i) = Tas(j, i) + dta;
      RHs(j+1, i) = RH(Pw(Was(j,i) + dwa), Ps(Tas(j,i) + dta));
    end
  end

  % plot desired results

  handle = 002;
  if plot_opts('Avg moisture')
    % show what the avg moisture content looks like
    plot_xy(handle, times/3600.0, mean(Ms, 1), 'Time (h)', 'Average malt moisture content');
  end

  handle = handle + 1;
  if plot_opts('Beta glucanase activity')
    % beta glucanase activity
    plot_xy(handle, times(1:length(times)-1)/3600.0, beta_gluc_profile(Tas(1,:)+273, mean(Ms, 1), times), 'Time (h)', '$\beta$ - Glucanase activity (BGU)');
  end

  handle = handle + 1;
  if plot_opts('Alpha amylase activity')
    % alpha amylase activity
    plot_xy(handle, times(1:length(times)-1)/3600.0, alpha_am_profile(Tas(1,:)+273, times), 'Time (h)', '$\alpha$ - Amylase activity (DU/g dm)')
  end

  handle = handle + 1;
  if plot_opts('Diastatic power')
    % diastatic power
    plot_xy(handle, times(1:length(times)-1)/3600.0, dias_pow_profile(Tas(1,:)+273, mean(Ms, 1), times), 'Time (h)', 'Diastatic power (WK/ 100 g dm)')
  end

  handle = handle + 1;
  if plot_opts('Limit dextrinase')
    % limit dextrinase
    plot_xy(handle, times(1:length(times)-1)/3600.0, limit_dextrinase_profile(Tas(1,:)+273, mean(Ms, 1), times), 'Time (h)', 'Limit-dextrinase (RPU/ 100 g dm)')
  end

  handle = handle + 1;
  if plot_opts('Malt temp profiles')
    % plot average first
    fig = figure(handle);
    ax = gca;
    plot(ax, (1:(nts+1))*dt/3600.0, mean(Tms, 1), 'DisplayName', 'average', 'Color', 'r', 'Linewidth', 2.0);
    % malt temperature evolution at various depths
    plot_xy_slices(handle, (1:(nts+1))*dt/3600.0, Tms, 'Time (h)', 'Malt temperature ($\circ$C)', 5, dz)
  end

  handle = handle + 1;
  if plot_opts('Air temp profiles')
    % plot average first
    fig = figure(handle);
    ax = gca;
    plot(ax, (1:nts)*dt/3600.0, mean(Tas, 1), 'DisplayName', 'average', 'Color', 'r', 'Linewidth', 2.0);
    % air temperature evolution at various depths
    plot_xy_slices(handle, (1:nts)*dt/3600.0, Tas, 'Time (h)', 'Air temperature ($\circ$C)', 5, dz)
  end

  handle = handle + 1;
  if plot_opts('Air RH profiles')
    % plot average first
    fig = figure(handle);
    ax = gca;
    plot(ax, (1:(nts))*dt/3600.0, mean(RHs, 1), 'DisplayName', 'average', 'Color', 'r', 'Linewidth', 2.0);
    % air moisture evolution at various depths
    plot_xy_slices(handle, (1:nts)*dt/3600.0, RHs, 'Time (h)', 'Air RH (kg water/kg dry air)', 5, dz)
  end

  handle = handle + 1;
  if plot_opts('Malt moisture profiles')
    % plot average first
    fig = figure(handle);
    ax = gca;
    plot(ax, (1:(nts+1))*dt/3600.0, mean(Ms, 1), 'DisplayName', 'average', 'Color', 'r', 'Linewidth', 2.0);
    % malt moisture evolution at various depths
    plot_xy_slices(handle, (1:(nts+1))*dt/3600.0, Ms, 'Time (h)', 'Malt moisture content (kg water/kg dry air)', 5, dz)
  end

end

function [G_in, Ta_in, RH_in, recirc] = parse_air_prog(air_prog, t)
  i = 1;
  air_prog_dims = size(air_prog);
  while i <= air_prog_dims(1) && t >= 3600*air_prog(i, 1)
    i = i + 1;
  end
  out_cell = num2cell(air_prog(i-1, 2:air_prog_dims(2)));
  [G_in, Ta_in, RH_in, recirc] = out_cell{:};
  recirc = recirc/100.0;
  RH_in = RH_in/100.0;
end

function plot_xy(fig_handle, xdata, ydata, xlab, ylab)
  fig = figure(fig_handle);
  ax = gca;

  plot(ax, xdata, ydata);

  xlim(ax, [0, max(xdata)]);
  xlabel(ax, xlab);
  ylabel(ax, ylab, 'Interpreter', 'latex');
  figure(fig_handle);
end

function plot_xy_slices(fig_handle, xdata, ydata, xlab, ylab, nslices, dz)
  fig = figure(fig_handle);
  ax = gca;
  hold(ax);
  c = ['r'; 'b'; 'k'; 'g'; 'c'; 'm'; 'y'];
  markers = ['o', '+', '*', '.', 's', 'v', 'p'];
  y_dims = size(ydata);
  nzs = y_dims(1);
  nslices = nslices - 1;
  zspacing = floor(1.0*nzs/nslices);

  % plot evenly-space, (nslices-2) profiles, then add profile at z=0 and z=top
  plot(ax, xdata, ydata(1,:), 'DisplayName', 'depth: 0 m (bottom)', 'Color', c(nslices+1), 'Marker', markers(nslices+1));
  for i = 1:nslices-1
    plot(ax, xdata, ydata(zspacing*i,:), 'DisplayName', ['depth: ', num2str(zspacing*i*dz), ' m'], 'Color', c(i), 'Marker', markers(i));
  end
  plot(ax, xdata, ydata(nzs,:), 'DisplayName', ['depth: ', num2str(nzs*dz), ' m (top)'], 'Color', c(nslices+2), 'Marker', markers(nslices+2));

  xlabel(ax, xlab);
  ylabel(ax, ylab, 'Interpreter', 'latex');
  xlim(ax, [0, max(xdata)]);
  legend(ax, 'show', 'Location', 'northwest');
  figure(fig_handle);
end

% change in malt temperature
% Bala's thesis, pg. 93
function [dtg] = dTm(Wa, dWa, dM, Ta, Cp_malt, M, Cp_water, Cp_watervapor, dz, rho, G, dt, h, Tm, rho_malt)
  % enthalpy of vaporization of water (kJ/ kg) 
  % assumed constant, taken at 25 C
  L_water = 1000*43.99/18;
  % kJ/min m^3 K
  % Bala's thesis, pg. 44
  hcv = 49.32*power(G, 0.6906)*60;
  % specific heat of malt (kJ/kg K)
  % Bala's thesis, pg. 23
  L_malt = L_water*(1 + 0.5904*exp(-0.1367*M));
  % specific heat of air (kg/kg K)
  % assumed constant, taken at 45 C from http://www.engineeringtoolbox.com/air-properties-d_156.html
  Cp_air = 1006;
  % helper variables
  % Bala's thesis, pg. 93
  A = 2*(Ta - Tm);
  B = Cp_malt + Cp_water*M;
  F = Cp_watervapor*Ta + L_water - Cp_water*Tm;
  Y = L_malt + Cp_watervapor*Ta - Cp_water*Tm;
  E = Cp_air + Cp_watervapor*(Wa - rho_malt*dz*dM/(G*dt));
  num = A + rho_malt*dM*(2*Y/hcv + dz*F/(G*E))/dt;
  denom = 1 + rho_malt*(2*B/hcv + dz*(B + Cp_water*dM)/(G*E))/dt;
  % change in malt temperature
  % Bala's thesis, pg. 93
  dtg = num/denom;
end

% change in air temperature
% Bala's thesis, pg. 92-93
function [dta] = dTa(Tm, dtm, rho_malt, h, dt, dM, Cp_malt, Cp_water, M, Cp_watervapor, Ta, G, Wa, dz)
  % enthalpy of vaporization of water (kJ/ kg) 
  % assumed constant, taken at 25 C
  L_water = 1000*43.99/18;
  % kJ/min m^3 K
  % Bala's thesis, pg. 44
  hcv = 49.32*power(G, 0.6906)*60;
  % specific heat of malt (kJ/kg K)
  % Bala's thesis, pg. 23
  L_malt = L_water*(1 + 0.5904*exp(-0.1367*M));
  % specific heat of air (kg/kg K)
  % assumed constant, taken at 45 C from http://www.engineeringtoolbox.com/air-properties-d_156.html
  Cp_air = 1006;
  % helper variables
  % Bala's thesis, pg. 93
  A = 2*(Ta - Tm);
  B = Cp_malt + Cp_water*M;
  F = Cp_watervapor*Ta + L_water - Cp_water*Tm;
  Y = L_malt + Cp_watervapor*Ta - Cp_water*Tm;
  E = Cp_air + Cp_watervapor*(Wa - rho_malt*dz*dM/(G*dt));
  % change in air temperature
  % Bala's thesis, pg. 93
  dta = -rho_malt*dz*(dtm*(B + Cp_water*dM) - dM*F)/(G*E*dt);
end

% partial pressure water, Pa
% given air moisture content, kg water/kg dry air
function [pw] = Pw(Wa)
  % molecular weight of water, g/mol
  MW_water = 18.0;
  % molecular weight of air, g/mol
  MW_air = 28.96;
  % atmospheric pressure, Pa
  P = 101325;
  % moles water/moles dry air
  Na_dry = Wa*MW_air/MW_water;
  % definition of partial pressure:
  % pressure * mole fraction water in air, Pa
  pw = P*Na_dry/(1.0 + Na_dry);
end

% air moisture content, kg water/kg dry air
% given relative humidity and saturation pressure
function [wa] = Wa_from_rh(RH, Ps)
  % molecular weight of water, g/mol
  MW_water = 18.0;
  % molecular weight of air, g/mol
  MW_air = 28.96;
  % atmospheric pressure, Pa
  P = 101325;
  % partial pressure water, Pa
  % by definition
  pw = RH*Ps;
  % solving for Wa in Na_dry and pw equations in function [pw] = Pw(Wa)
  wa = pw/(P-pw)*(MW_water/MW_air);
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
  % UNITS of k_alpha0??
  % PAPER'S VALUE IS 5.7654e9
  % BUT THIS APPEARS TO BE ORDERS OF
  % MAGNITUDE OFF. THE CURRENT VALUE
  % of 5.7654e7 PRODUCES RESULTS
  % THAT CLOSELY MATCH THE PAPER'S
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

% returns diastatic power profile based
% on the drying outputs: air temp, air moisture
% and their corresponding times
function [dp_prof] = dias_pow_profile(Tas, Ms, times)
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

% returns limit-dextrinase profile based
% on the drying outputs: air temp, air moisture
% and their corresponding times
function [ld_prof] = limit_dextrinase_profile(Tas, Ms, times)
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

% dext_prime, 1/s
% model parameter, 1/min
% pg. 161
function [ld_f] = lim_dext_f(T, M, dext)
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
