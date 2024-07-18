import math
import numpy as np
from scipy.optimize import root

from bdf import bdf


# state is (\xi, u, \eta, v, \lambda)


def f(time):
    return (9.81 + 0.05 * math.sin(2 * math.pi * time)) / 5


def f_der(time):
    return (0.05 * 2 * math.pi * math.cos(2 * math.pi * time)) / 5


def rhs(time, state):
    return np.array([state[1], -state[4] * state[0],
                     state[3], -state[4] * state[2] - f(time)])


def rhs_ode(time, state):
    return np.array([state[1], -state[4] * state[0],
                     state[3], -state[4] * state[2] - f(time),
                     (1 / (state[0] ** 2 + state[2] ** 2)) * (
                             (-f_der(time) * state[2] - 3 * f(time) * state[3]) - 4 * state[4] *
                             (state[0] * state[1] + state[2] * state[3]))])


def rhs_ode_stabilized(time, state):
    omega_sqr = 1
    return np.array([state[1], -state[4] * state[0],
                     state[3], -state[4] * state[2] - f(time),
                     (1 / (state[0] ** 2 + state[2] ** 2)) * (
                             (-f_der(time) * state[2] - 3 * f(time) * state[3]) + (omega_sqr * math.pi - 4 * state[4]) *
                             (state[0] * state[1] + state[2] * state[3])
                     )])


def rhs_ode_dumped(time, state):
    pho = state[0] ** 2 + state[2] ** 2
    gamma = 100
    omega_sqr = gamma * gamma
    return np.array([state[1], -state[4] * state[0],
                     state[3], -state[4] * state[2] - f(time),
                     (1 / pho) * (
                             ((omega_sqr / 2) - 2 * state[4]) * (state[0] * state[1] + state[2] * state[3]) +
                             (-f_der(time) * state[2] - 3 * f(time) * state[3]) +
                             2 * gamma * (state[1] ** 2 + state[3] ** 2 - f(time) * state[2] - state[4] * pho)
                     )
                     ])


def rhs_substituted(time, state):
    pho = state[0] ** 2 + state[2] ** 2
    gamma = 100
    omega_sqr = gamma * gamma
    l = (1 / pho) * (
            -f(time) * state[2] +
            (state[1] ** 2 + state[3] ** 2) +
            2 * gamma * (state[0] * state[1] + state[2] * state[3]) +
            (omega_sqr / 2) * (pho - 1)
    )
    return np.array([state[1], -l * state[0], state[3], -l * state[2] - f(time)])


def bdf_method(bdf, func, initial_ps, initial_time, timestep, n_iters, args=()):
    def F(state_, previous_state, time_):
        # differential part
        result_ = bdf(state_[0:4], *previous_state) - timestep * func(time_, state_)
        algebraic_part = [state_[0] ** 2 + state_[2] ** 2 - 1]
        result_ = np.append(result_, algebraic_part)
        return result_

    result = []

    previous_ps = initial_ps

    time = initial_time

    initial_guess = np.append(initial_ps[-1], 0)

    for i in range(n_iters):
        state = root(F, initial_guess, args=(previous_ps, time), method='lm', tol=1e-4)['x']
        # print(state)
        previous_ps = previous_ps[1:] + [state[0:4]]
        time += timestep
        result.append(state)
        initial_guess = np.array(result[-1])

    return np.array(result)


def setup_bdf(func, initial_p, initial_time, timestep, n_iters, args=()):
    def F(state_, previous_state, time_, bdf):
        # differential part
        result_ = bdf(state_[0:4], *previous_state) - timestep * func(time_, state_)
        algebraic_part = [state_[0] ** 2 + state_[2] ** 2 - 1]
        result_ = np.append(result_, algebraic_part)
        return result_

    if n_iters > 6:
        return None

    result = []

    previous_ps = [initial_p]

    time = initial_time

    initial_guess = np.append(initial_p, 0)
    for i in range(n_iters):
        state = root(F, initial_guess, args=(previous_ps, time, bdf(i + 1)), method='lm')['x']
        previous_ps += [state[0:4]]
        time += timestep
        result.append(state)
        initial_guess = np.array(result[-1])

    return np.array(result)


def implicit_euler(func, initial_p, initial_time, timestep, n_iters, args=()):
    def F(state_, previous_state, time_):
        # differential part (bdf of 1st order)
        result_ = state_[0:4] - previous_state - timestep * func(time_, state_)
        algebraic_part = [state_[0] ** 2 + state_[2] ** 2 - 1]
        result_ = np.append(result_, algebraic_part)
        return result_

    result = []

    previous_p = initial_p

    time = initial_time

    initial_guess = np.append(initial_p, 0)
    for i in range(n_iters):
        state = root(F, initial_guess, args=(previous_p, time), method='lm')
        previous_p = state[0:4]
        time += timestep
        result.append(state)
        initial_guess = np.array(result[-1])

    return np.array(result)


if __name__ == '__main__':
    print('Make sure you do all right')
