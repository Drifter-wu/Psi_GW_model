#初始质量为10M_sun 自旋为0的两个黑洞并和

import matplotlib.pyplot as plt
import numpy as np

def BOB_GW(Omega_0, phi_0, gamma, omega_0):
    M_sun = 2*10**33
    G = 6.6720*10**(-8)
    c = 3*10**10
    M = 10*M_sun

    unit_t = G*M/c**3
    unit_l = G*M/c**2


    def sech(value):
        a = 2/(np.exp(value)+np.exp(-value))
        return a


    t_p = -4.596304343924184    #peak time
    X = 3.719162155261737e-21         #peak mangnitude

    t_0 = 1.2

    tau = 1/gamma
    Omega_QNM = omega_0
    #########################################

    t = np.linspace(-160,60,916)

    k = (Omega_QNM**4-Omega_0**4)/(1-np.tanh((t_0-t_p)/tau))
    Omega = (Omega_0**4+k*(np.tanh((t-t_p)/tau) - np.tanh((t_0-t_p)/tau)))**(1/4)

    kar_plus =  (Omega_0**4+k*(1 - np.tanh((t_0-t_p)/tau)))**(1/4)
    kar_minu =  (Omega_0**4-k*(1 + np.tanh((t_0-t_p)/tau)))**(1/4)

    phi_22 = kar_plus*tau*(np.arctan(Omega/kar_plus) - np.arctan(Omega_0/kar_plus))+\
          kar_minu*tau*(np.arctan(Omega/kar_minu) - np.arctan(Omega_0/kar_minu))+\
          kar_minu*tau*(np.arctanh(Omega/kar_minu) - np.arctanh(Omega_0/kar_minu))-phi_0


    h_22 = X*sech(gamma*(t-t_p))*np.exp(-1j*phi_22)
    x_stor = 2*unit_t*(t-t_p)
    y_stor = -unit_l*h_22.real

    return -2*unit_t*(t-t_p), h_22.real

Omega_0 = 0.759  
phi_0 = 1.02679   


wR = 0.7146
wI = 0.058475
GW = BOB_GW(Omega_0, phi_0, wI, wR);
n = len(GW[0])
n_beg = 0
n_end = 660

np.savetxt('ringdown.txt', np.column_stack((GW[0][n_beg: n_end], GW[1][n_beg: n_end])))