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
    

def dry_malt(dt, tfinal, dz, z, air_program):
    # barley density, kg/m^3
    rho_barley = 600.0
    # dry barley heat capacity, J/(kg K)
    Cp_barley = 1300
    # specific heat of water vapor  J/(kg K)
    Cp_watervapor = 1870
    # specific heat of water  J/(kg K)
    Cp_water = 4180
    G = 310/3600.0
    
    nzs = int(z/dz)
    nts = int(tfinal/dt)
    print nzs, nts

    Tgs = np.empty((nzs, nts))
    Ms = np.empty((nzs, nts))
    Tas = np.empty((nzs+1, nts))
    Was = np.empty((nzs+1, nts))

    Tg_init = 20
    Tg_current = Tg_init*np.ones(nzs)

    # define smaller functions

    # drying rate paramater, 1/s
    k = lambda Ta: 139.3*np.exp(-4426/(Ta+273))
    # equilibrium moisture content of barley
    M_eq = lambda Ta, Wa: np.power(Wa, 0.5)*7040/np.power(1.8*Ta+32, 2) + 0.06015
    # change in moisture content of in slice
    # during time change "dt"
    dM = lambda Ta, M, Wa, dt: -k(Ta)*(M-M_eq(Ta, Wa))*dt/(1+k(Ta)*dt/2)
    # change in air moisture content over slice
    dWa = lambda rho, dz, dM, G, dt: rho*dz*dM/(G*dt)
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
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ADD RELATIVE HUMIDITY CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ADD RELATIVE HUMIDITY CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ADD RELATIVE HUMIDITY CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ADD RELATIVE HUMIDITY CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ADD RELATIVE HUMIDITY CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            dm = dM(Tas[j, i], Ms[j, i], Was[j, i], dt)
            dwa = dWa(rho_barley, dz, dm, G, dt)
            dtg = dTg(Was[j, i], dwa, dm, Tas[j, i], Cp_barley, Ms[j, i], Cp_water, Cp_watervapor, dz, rho_barley, G, dt, h_barley(G, Tas[j, i]), Tgs[j, i])
            dta = dTa(Tgs[j, i], dtg, rho_barley, h_barley(G, Tas[j, i]), dt, dm, Cp_barley, Cp_water, Ms[j, i], Cp_watervapor, Tas[j, i])
            Ms[j, i+1] = Ms[j, i] + dm
            Tgs[j, i+1] = Tgs[j, i] + dtg
            Was[j+1, i] = Was[j, i] + dwa
            Tas[j+1, i] = Tas[j, i] + dta
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_zslices = 5
    zspacing = int(nts/n_zslices)
    for i in range(n_zslices):
        ax.scatter(np.arange(nts)*dt/3600.0, Tg[zspacing*i, :], label="depth: " + str(zspacing*i) + " m")
    plt.show()




# change in temperature of malt in slice
def dTg(Wa, dWa, dM, Ta, Cp_grain, M, Cp_water, Cp_watervapor, dz, rho, G, dt, h, Tg):
    F = -dz*rho/(G*dt)
    numerator_i = Ta*(1005 + 1820*Wa) - Tg*F*(Cp_grain + M*Cp_water) - 2501000*dWa
    D = 2*rho/(h*dt)
    Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta
    numerator_ii = (1005+1820*(Wa+dWa))*(Ta + D*dM*Lv + D*Tg*(Cp_grain + M))/(1 + 1820*D*dM)
    denominator = (1005+1820*(Wa+dWa))*(D*(Cp_grain + (M+dM)*Cp_water) - F*(Cp_grain + (M+dM)*Cp_water))/(1 + 1820*D*dM)
    return (numerator_i + numerator_ii)/denominator

# change in air temperature across slice
def dTa(Tg, dTg, rho, h, dt, dM, Cp_grain, Cp_water, M, Cp_watervapor, Ta):
    D = 2*rho/(h*dt)
    Lv = 2501000 + 1820*Ta - Cp_watervapor*Ta
    return (D*(Tg+dTg)*(Cp_grain + (M+dM)*Cp_water) - D*Tg*(Cp_grain + M*Cp_water) - D*dM*Lv - Ta)/(1 + 1820*D*dM)


if __name__=="__main__":
    # change in height per iteration, m
    dz = 0.001
    # final height, m
    z = 0.3
    # change in time per iteration, s
    dt = 10
    # final time, s
    tfinal = 60*60*25
    dry_malt(dt, tfinal, dz, z, const_inflow)
