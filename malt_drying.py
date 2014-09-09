import sys
import numpy as np
import matplotlib.pyplot as plt

# returns the inlet air conditions
# at time "t", here simply given as constants
def const_inflow(t):
    # air inlet temperature, degC
    Ta_in = 60
    # air inlet moisture content, kg water/kg dry air
    Wa_in = 0.008
    return [Ta_in, Wa_in]
    
def lopez_program(t):
    hours_elapsed = t/float(3600)
    if hours_elapsed < 20:
        return [60, 0.008]
    elif hours_elapsed > 20 and hours_elapsed < 24:
        return [65, 0.008]
    else:
        return [80, 0.008]

def dry_malt(dt, tfinal, dz, z, air_program):
    # barley density, kg/m^3
    rho_barley = 600.0
    # dry barley heat capacity, kJ/(kg K)
    Cp_barley = 1.300
    # specific heat of water vapor kJ/(kg K)
    Cp_watervapor = 1.870
    # specific heat of water  kJ/(kg K)
    Cp_water = 4.180
    G = 310/3600.0
    
    nzs = int(z/dz)
    nts = int(tfinal/dt)
    tfinal = nts*dt

    Tgs = np.empty((nzs, nts+1))
    Ms = np.empty((nzs, nts+1))
    Tas = np.empty((nzs+1, nts))
    Was = np.empty((nzs+1, nts))

    Tg_init = 20
    Tgs[:, 0] = Tg_init*np.ones(nzs)
    
    Ms_init = 0.8
    Ms[:, 0] = Ms_init*np.ones(nzs)

    times = np.linspace(0, tfinal, nts+1)

    CONDENSATION_TOL = 1e-6
    maxiter = 100

    # define smaller functions

    # saturated water vapor pressure, Pa
    Ps = lambda Ta: 100000*np.exp(14.293  - 5291/(Ta + 273.15))/(3.2917 - 0.01527*(Ta + 273.15) + 2.54e-5*np.power(Ta + 273.15, 2))
    # relative humidity
    RH = lambda Pw, Ps: 100.0*Pw/Ps
    # drying rate paramater, 1/s
    k = lambda Ta: 139.3*np.exp(-4426/(Ta+273))
    # equilibrium moisture content of barley
    M_eq = lambda Ta, Wa: np.power(Wa, 0.5)*7040/np.power(1.8*Ta+32, 2) + 0.06015
    # change in moisture content of in slice
    # during time change "dt"
    dM = lambda Ta, M, Wa, dt: -k(Ta)*(M-M_eq(Ta, Wa))*dt/(1+k(Ta)*dt/2)
    # change in air moisture content over slice
    dWa = lambda rho, dz, dM, G, dt: -rho*dz*dM/(G*dt)
    # heat transfer coefficient, J/(K s m^3)
        # G - air flow rate, kg/(m^2 s)
        # Ta - air temp, degC
    h_barley = lambda G, Ta: 856800*np.power(G*(Ta+273)/101325, 0.6)
    for i in range(nts):
        t = i*dt
        Ta_in, Wa_in = air_program(t)
        Tas[0, i] = Ta_in
        Was[0, i] = Wa_in
        for j in range(nzs):
            dm = 0
            dwa = 0
            dm = dM(Tas[j, i], Ms[j, i], Was[j, i], dt)
            dwa = dWa(rho_barley, dz, dm, G, dt)
            dtg = deltaTg(Was[j, i], dwa, dm, Tas[j, i], Cp_barley, Ms[j, i], Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas[j, i]), Tgs[j, i], rho_barley) # Tg_next(Was[j, i], dwa, dm, Tas[j, i], Cp_barley, Ms[j, i], Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas[j, i]), Tgs[j, i]) - Tgs[j,i]
            dta = deltaTa(Tgs[j, i], dtg, rho_barley, h_barley(G, Tas[j, i]), dt, dm, Cp_barley, Cp_water, Ms[j, i], Cp_watervapor, Tas[j, i], G, Was[j,i], dz) # Ta_next(Tgs[j, i], dtg, rho_barley, h_barley(G, Tas[j, i]), dt, dm, Cp_barley, Cp_water, Ms[j, i], Cp_watervapor, Tas[j, i]) - Tas[j,i]
            if RH(Pw(Was[j,i] + dwa), Ps(Tas[j,i] + dta)) > 98:
                dm_s = dm
                ddm = dm_s/6.0
                # run condensation procedure as described in
                # Bala's thesis, pg. 95
                iters = 0
                while RH(Pw(Was[j,i] + dwa), Ps(Tas[j,i] + dta)) > 98 and iters < maxiter:
                    dm = dm - ddm
                    dwa = dWa(rho_barley, dz, dm, G, dt)
                    dtg = deltaTg(Was[j, i], dwa, dm, Tas[j, i], Cp_barley, Ms[j, i], Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas[j, i]), Tgs[j, i], rho_barley) # Tg_next(Was[j, i], dwa, dm, Tas[j, i], Cp_barley, Ms[j, i], Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas[j, i]), Tgs[j, i]) - Tgs[j,i]
                    dta = deltaTa(Tgs[j, i], dtg, rho_barley, h_barley(G, Tas[j, i]), dt, dm, Cp_barley, Cp_water, Ms[j, i], Cp_watervapor, Tas[j, i], G, Was[j,i], dz) # Ta_next(Tgs[j, i], dtg, rho_barley, h_barley(G, Tas[j, i]), dt, dm, Cp_barley, Cp_water, Ms[j, i], Cp_watervapor, Tas[j, i]) - Tas[j,i]
                    iters = iters + 1
                if iters == maxiter:
                    print 'search iterations exceeded'
                    quit()
                dm_l = dm
                dm_r = dm_s
                iters = 0
                while np.abs(RH(Pw(Was[j,i] + dwa), Ps(Tas[j,i] + dta)) - 98) > CONDENSATION_TOL and iters < maxiter:
                    if RH(Pw(Was[j,i] + dwa), Ps(Tas[j,i] + dta)) - 98 > 0:
                        dm_r = dm
                        dm = dm - (dm_r - dm_l)/2.0
                    else:
                        dm_l = dm
                        dm = dm + (dm_r - dm_l)/2.0
                    dwa = dWa(rho_barley, dz, dm, G, dt)
                    dtg = deltaTg(Was[j, i], dwa, dm, Tas[j, i], Cp_barley, Ms[j, i], Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas[j, i]), Tgs[j, i], rho_barley) # Tg_next(Was[j, i], dwa, dm, Tas[j, i], Cp_barley, Ms[j, i], Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas[j, i]), Tgs[j, i]) - Tgs[j,i]
                    dta = deltaTa(Tgs[j, i], dtg, rho_barley, h_barley(G, Tas[j, i]), dt, dm, Cp_barley, Cp_water, Ms[j, i], Cp_watervapor, Tas[j, i], G, Was[j,i], dz) # Ta_next(Tgs[j, i], dtg, rho_barley, h_barley(G, Tas[j, i]), dt, dm, Cp_barley, Cp_water, Ms[j, i], Cp_watervapor, Tas[j, i]) - Tas[j,i]
                    iters = iters + 1
                if iters == maxiter:
                    print 'bisection iterations exceeded'
                    quit()

                # print "At height:", j*dz, "m"
                # print "Ta+1 disparity:", dta - (Ta_next_check(dz, rho_barley, G, dt, h_barley(G, Tas[j,i]), Tas[j,i], Cp_watervapor, Was[j,i], Tgs[j,i], dtg, Cp_barley, Ms[j,i], dm, Cp_water, dwa) - Tas[j,i])
                # print "Ta:", Tas[j, i], "Tg:", Tgs[j, i], "Wa:", Was[j, i], "M:", Ms[j, i], "RH:", RH(Pw(Was[j,i]), Ps(Tas[j,i])), "Ps:", Ps(Tas[j,i]), "Pw:", Pw(Was[j,i])
                # print "dM:", dm, "dWa:", dwa, "dTg:", dtg, "dTa:", dta, "h:", h_barley(G, Tas[j, i])
                # raw_input("Press Enter to continue...")
            Ms[j, i+1] = Ms[j, i] + dm
            Tgs[j, i+1] = Tgs[j, i] + dtg
            Was[j+1, i] = Was[j, i] + dwa
            Tas[j+1, i] = Tas[j, i] + dta

    # show what the avg moisture content looks like
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times/3600.0, np.average(Ms, axis=0))
    ax.set_xlim(left=0)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Average malt moisture content')
    plt.show()

    bg_prof = beta_gluc_profile(Tas[0,:]+273, np.average(Ms, axis=0), times)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times[:-1]/3600.0, bg_prof)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$\beta - $' + 'Glucanase activity (BGU)')
    ax.set_xlim(left=0)
    plt.show()

    am_prof = alpha_am_profile(Tas[0,:]+273, times)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times[:-1]/3600.0, am_prof)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$\alpha - $' + 'Amylase activity (DU/g dm)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=20)
    plt.show()

    ds_prof = dias_pow_profile(Tas[0,:]+273, np.average(Ms, axis=0), times)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times[:-1]/3600.0, ds_prof)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Diastatic power (WK/ 100 g dm)')
    ax.set_xlim(left=0)
    # ax.set_ylim(bottom=200)
    plt.show()

    ld_prof = limit_dextrinase_profile(Tas[0,:]+273, np.average(Ms, axis=0), times)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(times[:-1]/3600.0, ld_prof)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Limit-dextrinase (RPU/ 100 g dm)')
    ax.set_xlim(left=0)
    # ax.set_ylim(bottom=200)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hold(True)
    # n_zslices = 5
    # c = ['r', 'b', 'k', 'g', 'c', 'm', 'y']
    # zspacing = int(nzs/n_zslices)
    # for i in range(n_zslices):
    #     ax.plot(np.arange(nts+1)*dt/3600.0, Tgs[zspacing*i, :], label="depth: " + str(zspacing*i*dz) + " m", c=c[i], lw=1)
    # ax.plot(np.arange(nts+1)*dt/3600.0, Tgs[-1, :], label="depth: " + str(zspacing*n_zslices*dz) + " m", c=c[n_zslices], lw=1)
    # ax.set_xlabel('Time (h)')
    # ax.set_ylabel('Grain temperature (' + r'$\degree$' + 'C)')
    # ax.set_xlim(left=0)
    # ax.legend(loc=4)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hold(True)
    # n_zslices = 5
    # for i in range(n_zslices):
    #     ax.plot(np.arange(nts)*dt/3600.0, Tas[zspacing*i, :], label="depth: " + str(zspacing*i*dz) + " m", c=c[i], lw=1)
    # ax.plot(np.arange(nts)*dt/3600.0, Tas[-1, :], label="depth: " + str(zspacing*n_zslices*dz) + " m", c=c[n_zslices], lw=1)
    # ax.set_xlabel('Time (h)')
    # ax.set_ylabel('Air temperature (' + r'$\degree$' + 'C)')
    # ax.set_xlim(left=0)
    # ax.legend(loc=4)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hold(True)
    # n_zslices = 5
    # for i in range(n_zslices):
    #     ax.plot(np.arange(nts)*dt/3600.0, Was[zspacing*i, :], label="depth: " + str(zspacing*i*dz) + " m", c=c[i], lw=1)
    # ax.plot(np.arange(nts)*dt/3600.0, Was[-1, :], label="depth: " + str(zspacing*n_zslices*dz) + " m", c=c[n_zslices], lw=1)
    # ax.set_xlabel('Time (h)')
    # ax.set_ylabel('Air moisture content (kg water/kg dry air)')
    # ax.set_xlim(left=0)
    # ax.legend(loc=1)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.hold(True)
    # n_zslices = 5
    # for i in range(n_zslices):
    #     ax.plot(np.arange(nts+1)*dt/3600.0, Ms[zspacing*i, :], label="depth: " + str(zspacing*i*dz) + " m", c=c[i], lw=1)
    # ax.plot(np.arange(nts+1)*dt/3600.0, Ms[-1, :], label="depth: " + str(zspacing*n_zslices*dz) + " m", c=c[n_zslices], lw=1)
    # ax.set_xlabel('Time (h)')
    # ax.set_ylabel('Grain moisture content (kg water/kg dry air)')
    # ax.set_xlim(left=0)
    # ax.legend(loc=1)
    # plt.show()



# change in temperature of malt in slice
def Tg_next(Wa, dWa, dM, Ta, Cp_grain, M, Cp_water, Cp_watervapor, dz, rho, G, dt, h, Tg):
    F = -dz*rho/(G*dt)
    numerator_i = Ta*(1005 + 1820*Wa) - Tg*F*(Cp_grain + M*Cp_water) - 2501000*dWa
    D = 2*rho/(h*dt)
    Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta
    numerator_ii = (1005+1820*(Wa+dWa))*(Ta + D*dM*Lv + D*Tg*(Cp_grain + M*Cp_water))/(1 + 1820*D*dM)
    denominator = (1005+1820*(Wa+dWa))*(D*(Cp_grain + (M+dM)*Cp_water) + 1)/(1 + 1820*D*dM) - F*(Cp_grain + (M+dM)*Cp_water)
    # print "D:", D, "Lv:", Lv, "Cp_grain:", Cp_grain, "Cp_water:", Cp_water, "some shit:", (Ta + D*dM*Lv + D*Tg*(Cp_grain + M*Cp_water))
    # print "num_i:", numerator_i, "num_ii:", numerator_ii, "denom:", denominator
    return (numerator_i + numerator_ii)/denominator

# change in air temperature across slice
def Ta_next(Tg, dTg, rho, h, dt, dM, Cp_grain, Cp_water, M, Cp_watervapor, Ta):
    D = 2*rho/(h*dt)
    Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta
    return ((Tg+dTg)*(D*(Cp_grain + (M+dM)*Cp_water) + 1) - D*Tg*(Cp_grain + M*Cp_water) - D*dM*Lv - Ta)/(1 + 1820*D*dM)

def deltaTg(Wa, dWa, dM, Ta, Cp_grain, M, Cp_water, Cp_watervapor, dz, rho, G, dt, h, Tg, rho_grain):
    # enthalpy of vaporization of water (kJ/ kg) (assumed constant, taken at 25 C)
    L_water = 1000*43.99/18
    # !!!!!!!!!!!!!!!!!!!!
    #    UNITS of hcv??
    # !!!!!!!!!!!!!!!!!!!!
    # either J/s m^3 K or kJ/min m^3 K
    # hcv = 49.32*1000*np.power(G, 0.6906)
    hcv = 49.32*np.power(G, 0.6906)*60
    # specific heat of malt (kJ/kg K) from thesis
    L_malt = L_water*(1 + 0.5904*np.exp(-0.1367*M))
    # Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta
    Cp_grain = 1600
    F = Cp_watervapor*Ta + L_water - Cp_water*Tg
    A = 2*(Ta - Tg)
    B = Cp_grain + Cp_water*M
    # latent heat of grain (J/kg) from thesis
    # Lg = 1000*Lv*(1 + 0.5904*np.exp(-0.1367*M))
    Y = L_malt + Cp_watervapor*Ta - Cp_water*Tg
    Cp_air = 1006
    E = Cp_air + Cp_watervapor*(Wa - rho_grain*dz*dM/(G*dt))
    F = Cp_watervapor*Ta + L_water - Cp_water*Tg
    num = A + rho_grain*dM*(2*Y/hcv + dz*F/(G*E))/dt
    denom = 1 + rho_grain*(2*B/hcv + dz*(B + Cp_water*dM)/(G*E))/dt
    return num/denom

def deltaTa(Tg, dTg, rho_grain, h, dt, dM, Cp_grain, Cp_water, M, Cp_watervapor, Ta, G, Wa, dz):
    # enthalpy of vaporization of water (kJ/ kg) (assumed constant, taken at 25 C)
    L_water = 1000*43.99/18
    # !!!!!!!!!!!!!!!!!!!!
    #    UNITS of hcv??
    # !!!!!!!!!!!!!!!!!!!!
    # either J/s m^3 K or kJ/min m^3 K
    # hcv = 49.32*1000*np.power(G, 0.6906)
    hcv = 49.32*np.power(G, 0.6906)*60
    # specific heat of malt (kJ/kg K) from thesis
    L_malt = L_water*(1 + 0.5904*np.exp(-0.1367*M))
    # Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta
    Cp_grain = 1600
    F = Cp_watervapor*Ta + L_water - Cp_water*Tg
    A = 2*(Ta - Tg)
    B = Cp_grain + Cp_water*M
    # latent heat of grain (J/kg) from thesis
    # Lg = 1000*Lv*(1 + 0.5904*np.exp(-0.1367*M))
    Y = L_malt + Cp_watervapor*Ta - Cp_water*Tg
    Cp_air = 1006
    E = Cp_air + Cp_watervapor*(Wa - rho_grain*dz*dM/(G*dt))
    F = Cp_watervapor*Ta + L_water - Cp_water*Tg
    return -rho_grain*dz*(dTg*(B + Cp_water*dM) - dM*F)/(G*E*dt)


def Ta_next_check(dz, rho, G, dt, h, Ta, Cp_watervapor, Wa, Tg, dTg, Cp_grain, M, dM, Cp_water, dWa):
    F = -dz*rho/(G*dt)
    return (Ta*(1005 + 1820*Wa) +(Tg+dTg)*F*(Cp_grain + (M+dM)*Cp_water) - Tg*F*(Cp_grain + M*Cp_water) - 2501000*dWa)/(1005 + 1820*(Wa+dWa))
    
# partial pressure water, Pa
def Pw(Wa):
    MW_water = 18.0
    MW_air = 28.96
    P = 101325

    Na_dry = Wa*MW_air/MW_water
    return P*Na_dry/(1.0 + Na_dry)

    # Ma = Wa/(1.0+Wa)
    # Na = Ma*(MW_water/(Ma*MW_water + (1 - Ma)*MW_air))
    # return Na*P

def beta_gluc_profile(Tas, Ms, times, beta_init=700):
    # returns beta_glucanase activity profile based
    # on the drying outputs: air temp, air moisture
    # and their corresponding times
    # !!!!!!!!!!!!!!!!!!!!
    # uses RK4
    nsteps = times.shape[0]
    beta_glucs = np.empty(nsteps-1)
    beta_glucs[0] = beta_init
    # universal gas constant, J/mol K
    R = 8.3143
    # model parameter, pg 161, 1/min
    k_beta0 = 3.089e8
    # model parameter, pg 161, J/mol
    E_beta = 7.248e4
    k_beta = lambda T: k_beta0*np.exp(-E_beta/(R*T))
    # beta_prime, 1/s
    f = lambda T, Wa, beta: -k_beta(T)*Wa*beta/60.0
    # because there is no initial value assigned to
    # the air temperature or moisture in the bed,
    # there will be one less Ta and Wa than there are
    # times, so the indexing will differ by one
    for i in range(nsteps-2):
        dt = times[i+1] - times[i]
        k1 = f(Tas[i], Ms[i], beta_glucs[i])
        k2 = f((Tas[i] + Tas[i+1])/2.0, (Ms[i] + Ms[i+1])/2.0, beta_glucs[i] + dt*k1/2.0)
        k3 = f((Tas[i] + Tas[i+1])/2.0, (Ms[i] + Ms[i+1])/2.0, beta_glucs[i] + dt*k2/2.0)
        k4 = f(Tas[i+1], Ms[i+1], beta_glucs[i] + dt*k3)
        beta_glucs[i+1] = beta_glucs[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return beta_glucs

def alpha_am_profile(Tas, times, alpha_init=55):
    # returns alpha_amylase activity profile based
    # on the drying outputs: air temp, air moisture
    # and their corresponding times
    # !!!!!!!!!!!!!!!!!!!!
    # uses RK4 for integration
    nsteps = times.shape[0]
    alpha_ams = np.empty(nsteps-1)
    alpha_ams[0] = alpha_init
    # universal gas constant, J/mol K
    R = 8.3143
    # model parameter, pg 161, 1/min
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!
    #    UNITS of k_alpha0??
    # PAPER'S VALUE IS 5.7654e9
    # BUT THIS APPEARS TO BE ORDERS OF
    # MAGNITUDE OFF. THE CURRENT VALUE
    # of 5.7654e7 PRODUCES RESULTS
    # THAT CLOSELY MATCH THE PAPER'S
    # !!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    k_alpha0 = 5.7654e7
    # model parameter, pg 161, J/mol
    E_alpha = 7.8913e4
    k_alpha = lambda T: k_alpha0*np.exp(-E_alpha/(R*T))
    # alpha_prime, 1/s
    f = lambda T, alpha: -k_alpha(T)*alpha/60.0
    # because there is no initial value assigned to
    # the air temperature or moisture in the bed,
    # there will be one less Ta and Wa than there are
    # times, so the indexing will differ by one
    for i in range(nsteps-2):
        dt = times[i+1] - times[i]
        k1 = f(Tas[i], alpha_ams[i])
        k2 = f((Tas[i] + Tas[i+1])/2.0, alpha_ams[i] + dt*k1/2.0)
        k3 = f((Tas[i] + Tas[i+1])/2.0, alpha_ams[i] + dt*k2/2.0)
        k4 = f(Tas[i+1], alpha_ams[i] + dt*k3)
        alpha_ams[i+1] = alpha_ams[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return alpha_ams

def dias_pow_f(T, M, dias):
    # dias_prime, 1/s
    # model parameter, pg 161, 1/min
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!
    #    UNITS of k_dias0??
    # !!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    k_dias0 = 4.8037e10#e12
    # model parameter, pg 161, J/mol
    E_dias = 9.5056e4
    # model parameter, pg 161, units?
    r_dias = 5.0361e-2
    # model parameter, pg 161, WK/ 100 g dm
    K_dias = 470.0
    # model parameter, pg 161, % dry basis
    M_dias = .40
    # universal gas constant, J/mol K
    R = 8.3143
    k_dias = lambda T: k_dias0*np.exp(-E_dias/(R*T))
    # s, ?units?, must be 1/min
    s = lambda dias, T, M: r_dias*T*(M - M_dias)*(1-dias/K_dias)
    # only when M >= M_dias is s nonzero
    if M >= M_dias:
        return  (s(dias, T, M) - k_dias(T)*dias)/60.0
    else:
        return  -k_dias(T)*dias/60.0

def dias_pow_profile(Tas, Ms, times, dias_init=400):
    # returns diastatic power profile based
    # on the drying outputs: air temp, air moisture
    # and their corresponding times
    # !!!!!!!!!!!!!!!!!!!!
    # uses RK4 for integration
    nsteps = times.shape[0]
    dias_pows = np.empty(nsteps-1)
    dias_pows[0] = dias_init
    # because there is no initial value assigned to
    # the air temperature or moisture in the bed,
    # there will be one less Ta and Wa than there are
    # times, so the indexing will differ by one
    for i in range(nsteps-2):
        dt = times[i+1] - times[i]
        k1 = dias_pow_f(Tas[i], Ms[i], dias_pows[i])
        k2 = dias_pow_f((Tas[i] + Tas[i+1])/2.0, (Ms[i] + Ms[i+1])/2.0, dias_pows[i] + dt*k1/2.0)
        k3 = dias_pow_f((Tas[i] + Tas[i+1])/2.0, (Ms[i] + Ms[i+1])/2.0, dias_pows[i] + dt*k2/2.0)
        k4 = dias_pow_f(Tas[i+1], Ms[i+1], dias_pows[i] + dt*k3)
        dias_pows[i+1] = dias_pows[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return dias_pows 

def lim_dext_f(T, M, dext):
    # dext_prime, 1/s
    # model parameter, pg 161, 1/min
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!
    #    UNITS of k_dext0??
    # !!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    k_dext0 = 1.6554e18#e20
    # model parameter, pg 161, J/mol
    E_dext = 1.4516e5
    # model parameter, pg 161, units?
    r_dext = 6.4693
    # model parameter, pg 161, RPU/ 100 g dm
    K_dext = 93.963
    # model parameter, pg 161, % dry basis
    M_dext = 0.40
    # universal gas constant, J/mol K
    R = 8.3143
    k_dext = lambda T: k_dext0*np.exp(-E_dext/(R*T))
    # s, ?units?, must be 1/min
    s = lambda dext: r_dext*(1-dext/K_dext)
    # only when M >= M_dext is s nonzero
    if M >= M_dext:
        return  (s(dext) - k_dext(T)*dext)/60.0
    else:
        return  -k_dext(T)*dext/60.0

def limit_dextrinase_profile(Tas, Ms, times, dex_init=80):
    # returns limit-dextrinase profile based
    # on the drying outputs: air temp, air moisture
    # and their corresponding times
    # !!!!!!!!!!!!!!!!!!!!
    # uses RK4 for integration
    nsteps = times.shape[0]
    lim_dexts = np.empty(nsteps-1)
    lim_dexts[0] = dex_init
    # because there is no initial value assigned to
    # the air temperature or moisture in the bed,
    # there will be one less Ta and Wa than there are
    # times, so the indexing will differ by one
    for i in range(nsteps-2):
        dt = times[i+1] - times[i]
        k1 = lim_dext_f(Tas[i], Ms[i], lim_dexts[i])
        k2 = lim_dext_f((Tas[i] + Tas[i+1])/2.0, (Ms[i] + Ms[i+1])/2.0, lim_dexts[i] + dt*k1/2.0)
        k3 = lim_dext_f((Tas[i] + Tas[i+1])/2.0, (Ms[i] + Ms[i+1])/2.0, lim_dexts[i] + dt*k2/2.0)
        k4 = lim_dext_f(Tas[i+1], Ms[i+1], lim_dexts[i] + dt*k3)
        lim_dexts[i+1] = lim_dexts[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return lim_dexts 

def main():
    # change in height per iteration, m
    dz = 0.01
    # final height, m
    z = 0.3
    # change in time per iteration, s
    dt = 500
    # final time, s
    tfinal = 60*60*30
    dry_malt(dt, tfinal, dz, z, lopez_program)

if __name__=="__main__":
    main()
