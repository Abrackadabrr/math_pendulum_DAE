import numpy as np
from scipy.optimize import fsolve


def euler(func, state, time, timestep):
    return state + func(time, state) * timestep


def predict_correct(func, state, time, timestep):
    current_state = state
    state = euler(func, state, time, timestep)
    return current_state + func(time, state) * timestep


def hune(func, state, time, timestep):
    current_state = state
    state = euler(func, state, time, timestep)
    return ((current_state + func(time, state) * timestep) + state) * 0.5


def explicit_runge_kutta(table, ndim):
    def method(func, state, time, timestep):
        t = time + timestep * table.c
        k = np.zeros(ndim * table.c.shape[0]).reshape(table.c.shape[0], ndim)
        for i in range(table.c.shape[0]):
            k[i] = func(t[i], state + timestep * table.A[i] @ k)
        return state + timestep * table.b @ k

    return method


def implicit_runge_kutta(table, ndim):
    def function(k_value, func, state, t, timestep):
        result = np.array([])
        for i in range((table.c.shape[0])):
            result = np.concatenate(
                (result, func(t[i], state + timestep * table.A[i] @ (k_value.reshape(table.c.shape[0], ndim))))
            )
        return k_value - result

    def method(func, state, time, timestep):
        t = time + timestep * table.c
        k = np.zeros(ndim * table.c.shape[0])
        k = fsolve(function, k, args=(func, state, t, timestep))
        return state + timestep * table.b @ k.reshape(table.c.shape[0], ndim)

    return method


def integrator_method(next_step, func, initial_state, initial_time, timestep, n_iters, args=()):
    """
    Single step methods
    :param next_step: function determine method
    :param func: function determine physical model
    :param initial_state: initial state of system (Cauchy problem)
    :param timestep: time step
    :param n_iters: amount of iterations
    :param args: parameters to the func

    :return: sequence of approximations of system's state
    """
    result = []

    state = initial_state
    time = initial_time

    result.append(state)

    for i in range(n_iters):
        state = next_step(func, state, time, timestep, *args)
        time += timestep
        result.append(state)
    return np.array(result)


if __name__ == '__main__':
    print('Make sure you do all right')
