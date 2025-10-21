import numpy as np
import matplotlib.pylab as plt


def integrate_afo(phi0, omega0, omegaF, t_init, t_end, K=10000, lamb=1., save_dt=0.001, dt=0.00001):
    """ this function integrates an AFO from t_init to t_end
    it starts with initial conditions phi0 and omega0
    returns: numpy vectors t, phi and omega
    """
    # hoe many integration steps in the internal loop
    internal_step = int(round(save_dt / dt))

    # how many steps till t_end
    num_steps = int(round((t_end - t_init) / save_dt))

    # we preallocate memory
    t = np.zeros(num_steps + 1)
    phi = np.zeros_like(t)
    omega = np.zeros_like(t)

    # we set initial conditions
    t[0] = t_init
    omega[0] = omega0
    phi[0] = phi0

    # our temp variables
    phi_temp = phi0
    omega_temp = omega0
    t_temp = t_init

    # the main integration loop
    for i in range(num_steps):
        # internal integration loop
        for j in range(internal_step):
            pert = -K * np.sin(omegaF * t_temp) * np.sin(phi_temp)
            phi_temp += (lamb * omega_temp + pert) * dt
            omega_temp += pert * dt
            t_temp += dt

        # save data
        t[i + 1] = t_temp
        phi[i + 1] = phi_temp
        omega[i + 1] = omega_temp

    return t, phi, omega
K =  1000.
omegaF = 30
lamb = 1.

# we define the time resolution (save_dt) and the internal integration step dt
dt = 0.00001
save_dt = 0.001

# duration of the integration
t_end = 20.
t_init = 0.

phi0 = 0
omega0 = 100

# we integrate
t, phi, omega = integrate_afo(phi0, omega0, omegaF, t_init, t_end, K=10000, lamb=1., save_dt=0.001, dt=0.00001)

# now we plot the results
plt.figure()
plt.subplot(2,1,1)
plt.plot(t, phi)
plt.ylabel(r'$\phi$')
plt.subplot(2,1,2)
plt.plot(t, lamb * omega)
plt.plot([t_init, t_end],[omegaF, omegaF], '--k')
plt.ylabel(r'$\omega$')
plt.xlabel('Time [s]')
plt.show()