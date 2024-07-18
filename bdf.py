def bdf1(y_np1, y_n):
    return y_np1 - y_n


def bdf2(y_np1, y_n1, y_n):
    return (3 * y_np1 / 2) - 2 * y_n + y_n1 / 2


def bdf3(y_np1, y_n2, y_n1, y_n):
    return (11 * y_np1 / 6) - 3 * y_n + (3 * y_n1 / 2) - y_n2 / 3


def bdf4(y_np1, y_n3, y_n2, y_n1, y_n):
    return (25 * y_np1 / 12) - 4 * y_n + 3 * y_n1 - 4 * y_n2 / 3 + 1 / 4 * y_n3


def bdf5(y_np1, y_n4, y_n3, y_n2, y_n1, y_n):
    return (137 * y_np1 / 60) - 5 * y_n + 5 * y_n1 - (10 * y_n2 / 3) + (5 * y_n3 / 4) - y_n4 / 5


def bdf6(y_np1, y_n5, y_n4, y_n3, y_n2, y_n1, y_n):
    return (147 * y_np1 / 60) - 6 * y_n + (15 * y_n1 / 2) - (20 * y_n2 / 3) + (15 * y_n3 / 4) - \
           (6 * y_n4 / 5) + y_n5 / 6


def bdf(bdf_order):
    if bdf_order == 1:
        return bdf1
    if bdf_order == 2:
        return bdf2
    if bdf_order == 3:
        return bdf3
    if bdf_order == 4:
        return bdf4
    if bdf_order == 5:
        return bdf5
    if bdf_order == 6:
        return bdf6
