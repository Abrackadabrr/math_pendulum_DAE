import integrator
import numpy as np
import matplotlib.pyplot as plt

from butcher_table import rk4, HummerHill
from dae import f, rhs_ode, rhs_ode_stabilized, rhs_ode_dumped, rhs_substituted

xi_0 = 3 / 5
xi_der_0 = -4 / 25
eta_0 = -4 / 5
eta_der_0 = -3 / 25
lambda_0 = -f(0) * eta_0 + (xi_der_0) ** 2 + (eta_der_0) ** 2

initial_conditions = np.array([xi_0, xi_der_0, eta_0, eta_der_0, lambda_0])
timestep = 0.001
fulltime = 100
timegrid = np.arange(0, fulltime + timestep, timestep)

result = integrator.integrator_method(integrator.explicit_runge_kutta(rk4(), initial_conditions.shape[0]),
                                      rhs_ode,
                                      initial_conditions, 0,
                                      timestep,
                                      timegrid.shape[0] - 1)

r = np.square(result[:, 0]) + np.square(result[:, 2]) - 1
r_but_sqrt = np.sqrt(np.square(result[:, 0]) + np.square(result[:, 2])) - 1

# plotting
fig, ax = plt.subplots(ncols=1)
# k = np.polyfit(np.log(timegrid[10:]), np.log(np.abs(r[10:])), 1)
ax.grid()
ax.plot(timegrid, r_but_sqrt, 'b.-')
ax.ticklabel_format(axis='both', style="sci", useMathText=True)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()
ax.set_ylabel(r"$\sqrt{\xi^2 + \eta^2} - 1$")
ax.set_xlabel(r"$t$")
# /media/evgen/SecondLinuxDisk/Interviews/Rythm
# ax[1].grid()
# ax[1].ticklabel_format(axis='both', style='scientific', useMathText=True)
# ax[1].plot(timegrid, r_but_sqrt, 'b.-')
# ax[1].set_ylabel(r"$\sqrt{\xi^2 + \eta^2} - 1$")
# ax[1].set_xlabel(r"$t$")
plt.show()

# fig, ax = plt.subplots()
# ax.plot(timegrid, result[:, -1], 'b.-')
# ax.grid()
# ax.set_title("lambda value")

plt.show()
