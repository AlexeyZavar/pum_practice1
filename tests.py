def dummy1(a):
    return True


def dummy2(a):
    return False


def dummy3(a, b, c, d):
    return True


def simple1(a):
    return not a


def simple2(a, b):
    return a or b


def simple3(a, b, c, d):
    return a and b or not c and d


def simple4(a, b, c, abc):
    return a and b or not c and not abc


def simple5(a, b, c):
    return a or b and c


def simple6(a):
    return a and not a


def medium1(a, b):
    return a and b or (b and not a)


def medium2(a, b, c, d, e):
    return a and b or (b and not a) or (c and e) or (not c and e) or (c and not e) or (not c and not e)


def medium3(a, b, c, d, e):
    return (a and b) and (b and not a) or (c and e) and (not c and e) or (c and not e) or (not c and not e)


def medium4(a, b, c, d, e, f):
    return a and not b or c and not d and e and f


def medium5(a, b, c, d, e, f): return (a and not b and not d) and (not f or e or not c and f) or (
        a and not b and not c and d) or (a and not b and c and d) or (c and not d and e and f) and (c or not e)


def ultimate1(a, b, c, d, e, f, g, h, j, k, i, o, p, u, y, t, r, z, v, n, m, s):
    return a and b or not c and not d and e and f or g and h and j or not k and i


def ege1(x, y, z, w):
    return (w or y) and (y or not x) and (z or not y)


def denchik1(a, b, c):
    return a and (b or c)


def wiki1(x1, x2, x3):
    return x2 and not x3 or x1


def wiki2(x1, x2, x3, x4):
    return (not x1 and not x2 and not x3) or (x1 and x2 and x3) or (not x1 and x3 and not x4)


def wiki3(x, y, z, t):
    return (y and not z and not t) or (x and not y) or (x and z)


def long_running1(x, y, z, t, e):
    return (y and not z and not t) or (x and not y) or (x and z) or e
