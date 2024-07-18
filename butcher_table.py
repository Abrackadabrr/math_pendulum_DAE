import numpy as np


class ButherTable:
    def __init__(self, a, b, c):
        self.A = a
        self.b = b
        self.c = c


def rk4():
    A = np.array([[0, 0, 0, 0],
                  [1/2, 0, 0, 0],
                  [0, 1/2, 0, 0],
                  [0, 0, 1, 0]])
    c = np.array([0, 1/2, 1/2, 1])
    b = np.array([1/3, 1/6, 1/6, 1/3])
    return ButherTable(A, b, c)


def HummerHill():
    value = np.sqrt(3) / 6
    A = np.array([[1/4, 1/4 - value],
                  [1/4 + value, 1/4]])
    c = np.array([1/2 - value, 1/2 + value])
    b = np.array([1/2, 1/2])
    return ButherTable(A, b, c)


if __name__ == '__main__':
    print('Make sure you do all right')
