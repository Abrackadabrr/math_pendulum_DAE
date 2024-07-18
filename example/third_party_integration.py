import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

from dae import f, rhs_ode, rhs_ode_stabilized, rhs_ode_dumped


def solout(t, y):
    global result
    result.append([t, *y])


xi_0 = 3 / 5
xi_der_0 = -4 / 25
eta_0 = -4 / 5
eta_der_0 = -3 / 25
lambda_0 = -f(0) * eta_0 + xi_der_0 ** 2 + eta_der_0 ** 2

initial_conditions = np.array([xi_0, xi_der_0, eta_0, eta_der_0, lambda_0])
timestep = 0.5
fulltime = 100

result = []

solver = ode(rhs_ode).set_integrator('dopri5', rtol=1e-7, first_step=timestep, nsteps=1e8)
solver.set_solout(solout)
solver.set_initial_value(initial_conditions, 0)
solver.integrate(fulltime)

result = np.array(result)
residual = np.square(result[:,1]) + np.square(result[:,3]) - 1

avarage_step = np.mean(result[1:,0] - result[:-1, 0])
print(avarage_step)

# plotting
fig, ax = plt.subplots(ncols=2)
print(np.polyfit(np.log(result[1:,0]), np.log(np.abs(residual[1:])), 1))
ax[0].grid()
ax[0].plot(result[:,0], residual, 'b.-')
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
ax[0].set_title("residual value")
ax[1].grid()
ax[1].plot(result[:, 0], result[:, 1], 'b.-')
ax[1].set_title("xi")
# ax[1].set_aspect('equal')

plt.show()
