from sympy import Function, Symbol
from sympy.printing.latex import latex
from sympy import solve

t = Symbol('t')
xi = Function('\mu')(t)
eta = Function('\eta')(t)
f = Function('f')(t)

gamma = Symbol('\gamma')
omega = Symbol('\omega')
l = Symbol('\lambda')
u = Symbol('u')
v = Symbol('v')

r = eta**2 + xi**2 - 1
baumgarthe = (r.diff(t).diff(t) + 2 * gamma * r.diff(t) + omega**2 * r) / 2
baumgarthe = baumgarthe.subs(xi.diff(t).diff(t), -l * xi)
baumgarthe = baumgarthe.subs(xi.diff(t), u)
baumgarthe = baumgarthe.subs(eta.diff(t).diff(t), -l * eta - f)
baumgarthe = baumgarthe.subs(eta.diff(t), v).simplify()
print(latex(solve((baumgarthe, 0), l)[l].simplify()))