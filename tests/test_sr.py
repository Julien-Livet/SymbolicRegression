import math
import numpy as np
import operator
import random
import sr
import sympy

def sym_conv(x, y):
    return sympy.sympify("conv" + str(x) + ", " + str(y))

def test_x1_mul_x2():
    model = sr.SR(niterations = 3,
                  unary_operators = {"-": (operator.neg, operator.neg)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)
    #unary_operators = {"-": operator.neg,
    #                   "abs": (sympy.Abs, operator.abs),
    #                   "inv": (lambda x: 1 / x, lambda x: 1 / x),
    #                   "sqrt": (sympy.sqrt, np.sqrt),
    #                   "cos": (sympy.cos, np.cos),
    #                   "sin": (sympy.sin, np.sin),
    #                   "log": (sympy.log, np.log),
    #                   "exp": (sympy.exp, np.exp)}
    #binary_operators = {"+": (operator.add, operator.add),
    #                    "-": (operator.sub, operator.sub)
    #                    "*": (operator.mul, operator.mul),
    #                    "/": (operator.truediv, operator.truediv),
    #                    "//": (operator.floordiv, operator.floordiv),
    #                    "%": (operator.mod, operator.mod),
    #                    "conv": (sym_conv, sr.convolve),
    #                    "**": (sympy.Pow, operator.pow}

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = x1 * x2

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("x1*x2"))

def test_x1_add_x2():
    model = sr.SR(niterations = 3,
                  unary_operators = {"-": (operator.neg, operator.neg)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = x1 + x2

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("x1+x2"))

def test_x1_2_add_x2_2_sub_x1_mul_x2():
    model = sr.SR(niterations = 3,
                  unary_operators = {"-": (operator.neg, operator.neg)},
                  binary_operators = {"*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = (x1 - x2) ** 2 + x1 * x2

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.sympify("(x1-x2)**2+x1*x2"))))

def test_a_mul_x1_add_b():
    model = sr.SR(niterations = 3,
                  binary_operators = {"+": (operator.add, operator.add)},
                  foundBreak = True)

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    a = random.random()
    b = random.random()
    y = a * x1 + b

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    print(model.bestExpressions)
    assert(sr.expr_eq(model.bestExpressions[0][0], sympy.sympify(str(a) + "*x1+" + str(b))))

def test_a_mul_x2_add_b_mul_x2_add_c():
    model = sr.SR(niterations = 3,
                  binary_operators = {"+": (operator.add, operator.add)},
                  foundBreak = True)

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    a = random.random()
    b = random.random()
    c = random.random()
    y = a * x1 + b * x2 + c

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(model.bestExpressions[0][0], sympy.sympify(str(a) + "*x1+" + str(b) + "*x2+" + str(c))))

def test_pysr():
    X = 2 * np.random.randn(5, 100)
    y = 2.5382 * np.cos(X[3, :]) + X[0, :] ** 2 - 0.5

    model = sr.SR(niterations = 3,
                  unary_operators = {"cos": (sympy.cos, np.cos)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True,
                  eps = 1e-6)

    model.predict(X, y)

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(model.bestExpressions[0][0], sympy.sympify("2.5382 * cos(x3) + x0 ** 2 - 0.5")))

def test_sym_expr_eq():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = a*x + b
    expr2 = c*x + d

    assert(sr.sym_expr_eq(expr1, expr2, [x]))

    expr1 = a*x + b
    expr2 = c*y + d

    assert(not sr.sym_expr_eq(expr1, expr2, [x, y]))

def test_line():
    u = 2 * np.random.rand(2) - np.ones(2)
    u /= np.linalg.norm(u)
    
    p0 = 10 * np.random.rand(2) - 5 * np.ones(2)
    
    x = []
    y = []
    
    for i in range(0, 10):
        t = 10 * random.random() - 5
        p = t * u + p0
        x.append(p[0])
        y.append(p[1])
    
    x = np.array(x)
    y = np.array(y)

    #import matplotlib.pyplot as plt
    #plt.scatter(x, y)
    #plt.show()

    model = sr.SR(niterations = 3,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("a * x + b * y + c")],
                  avoided_expr = [sympy.sympify("0")],
                  binary_operators = {"+": (operator.add, operator.add)},
                  foundBreak = True,
                  epsloss = 1e-4)

    model.predict([x, y], np.zeros(len(x)), ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.sym_expr_eq(model.bestExpressions[0][0], sympy.sympify("a * x + b * y + c"), sympy.symbols("x y")))

def test_circle():
    p0 = 10 * np.random.rand(2) - 5 * np.ones(2)
    rho = 4
    
    x = []
    y = []
    
    for i in range(0, 10):
        theta = 2 * math.pi * random.random()
        x.append(p0[0] + rho * math.cos(theta))
        y.append(p0[1] + rho * math.sin(theta))

    x = np.array(x)
    y = np.array(y)

    #import matplotlib.pyplot as plt
    #ax = plt.gca()
    #ax.set_aspect('equal', adjustable = 'box')
    #plt.scatter(x, y)
    #plt.show()

    model = sr.SR(niterations = 2,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("(x - x0) ** 2 + (y - y0) ** 2 - R ** 2")],
                  avoided_expr = [sympy.sympify("0")],
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True,
                  epsloss = 1e-4)

    model.predict([x, y], np.zeros(len(x)), ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.sym_expr_eq(model.bestExpressions[0][0], sympy.sympify("(x - x0) ** 2 + (y - y0) ** 2 - R ** 2"), sympy.symbols("x y")))

def test_plane():
    n = 2 * np.random.rand(3) - np.ones(3)
    n /= np.linalg.norm(n)
    
    u = 2 * np.random.rand(3) - np.ones(3)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    u = np.cross(v, n)
    
    p0 = 10 * np.random.rand(3) - 5 * np.ones(3)
    
    x = []
    y = []
    z = []
    
    for i in range(0, 10):
        t1 = 10 * random.random() - 5
        t2 = 10 * random.random() - 5
        p = t1 * u + t2 * v + p0
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    model = sr.SR(niterations = 3,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("a * x + b * y + c * z + d")],
                  avoided_expr = [sympy.sympify("0")],
                  binary_operators = {"+": (operator.add, operator.add)},
                  foundBreak = True,
                  epsloss = 1e-4)

    model.predict([x, y, z], np.zeros(len(x)), ["x", "y", "z"])

    assert(sr.sym_expr_eq(model.bestExpressions[0][0], sympy.sympify("a * x + b * y + c * z + d"), sympy.symbols("x y z")))

def test_sphere():
    p0 = 10 * np.random.rand(3) - 5 * np.ones(3)
    rho = 4
    
    x = []
    y = []
    z = []
    
    for i in range(0, 10):
        theta = math.pi * random.random() - math.pi / 2
        phi = 2 * math.pi * random.random()
        x.append(p0[0] + rho * math.cos(theta) * math.cos(phi))
        y.append(p0[1] + rho * math.cos(theta) * math.sin(phi))
        z.append(p0[2] + rho * math.sin(theta))

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    model = sr.SR(niterations = 3,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("(x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - R ** 2")],
                  avoided_expr = [sympy.sympify("0")],
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True,
                  epsloss = 1e-4)

    model.predict([x, y, z], np.zeros(len(x)), ["x", "y", "z"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.sym_expr_eq(model.bestExpressions[0][0], sympy.sympify("(x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - R ** 2"), sympy.symbols("x y z")))

def test_gplearn():
    from sklearn.utils.random import check_random_state
    
    rng = check_random_state(0)

    X_train = np.transpose(rng.uniform(-1, 1, 100).reshape(50, 2))
    y_train = X_train[0, :]**2 - X_train[1, :]**2 + X_train[1, :] - 1
    
    model = sr.SR(niterations = 3,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)
    print(X_train)

    model.predict(X_train, y_train)

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("x0**2-x1**2+x1-1"))
