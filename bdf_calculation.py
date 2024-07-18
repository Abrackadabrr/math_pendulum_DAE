import numpy as np
import matplotlib.pyplot as plt

import dae
import bdf

# формирование начальных условий
xi_0 = 3 / 5
xi_der_0 = -4 / 25
eta_0 = -4 / 5
eta_der_0 = -3 / 25
initial_value = np.array([xi_0, xi_der_0, eta_0, eta_der_0])
timestep = 0.1
fulltime = 100
timegrid = np.arange(timestep, fulltime + timestep, timestep)

result = np.array([np.array(np.append(initial_value, 2))])

# разгон многошагового метода
bdf_order = 6
result = np.append(result, dae.setup_bdf(dae.rhs, initial_value, timegrid[1], timestep, bdf_order-1), axis=0)
# решение многошаговым методом
initial_p = [np.array(state[0:4]) for state in result]
result = np.append(result,
                   dae.bdf_method(bdf.bdf(bdf_order), dae.rhs, initial_p, timegrid[bdf_order], timestep,
                                  timegrid.shape[0] - bdf_order), axis=0)

# анализ полученных решений
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

fig, ax = plt.subplots()
ax.plot(timegrid, result[:, 0], 'b.-')
ax.grid()
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\xi$")
plt.show()

fig, ax = plt.subplots()
ax.plot(timegrid, result[:, -1], 'b.-')
ax.grid()
ax.set_title(r"$\lambda$ value")
plt.show()