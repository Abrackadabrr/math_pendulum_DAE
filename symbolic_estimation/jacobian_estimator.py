from sympy import Symbol, Matrix, simplify
from sympy.tensor.array import derive_by_array
from sympy.printing.latex import latex
from numpy import eye


eta = Symbol('eta')
xi = Symbol('xi')
u = Symbol('u')
v = Symbol('v')
l = Symbol('l')
f = Symbol('f')
omega_sqr = Symbol('w^2')
gamma = Symbol('gamma')
f_dot = Symbol('f_dot')


# 5th component of rhs in index reduction method
r = (1 / (xi ** 2 + eta ** 2)) * (
        (2 * omega_sqr - 4 * l) * (xi * u + eta * v) +
        (-f_dot * eta - 3 * f * v) +
        gamma * (
                u ** 2 + v ** 2 + f * eta -
                l * (xi ** 2 + eta ** 2)
        )
)

r = simplify(r)
gradient = simplify(derive_by_array(r, (xi, u, eta, v, l)))

J = Matrix([[0, 1, 0, 0, 0],
            [-l, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, -l, 0, 0],
            gradient])

eigenvalues = list(J.eigenvals().keys())
print(eigenvalues)

basis = (J - eigenvalues[-1]*eye(5)).nullspace()
print(basis)
