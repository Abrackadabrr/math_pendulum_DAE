from sympy import Function, Symbol
from sympy.printing.latex import latex

import bdf

y_np1 = Symbol('y_{n+1}')
y_n5 = Symbol('y_{n-5}')
y_n4 = Symbol('y_{n-4}')
y_n3 = Symbol('y_{n-3}')
y_n2 = Symbol('y_{n-2}')
y_n1 = Symbol('y_{n-1}')
y_n = Symbol('y_{n}')
h = Symbol('h')
f = Symbol('f_{n+1}')

args = [y_n5, y_n4, y_n3, y_n2, y_n1, y_n]

for i in range(4):
    print(latex(bdf.bdf(i+1)(y_np1, *(args[5-i:]))), sep='\\' + '\n')