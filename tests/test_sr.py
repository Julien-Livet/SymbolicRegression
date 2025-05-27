import math
import numpy as np
import operator
import random
import sr
import sympy

#Commented tests fail and need some work

def sym_conv(x, y):
    return sympy.sympify("conv" + str(x) + ", " + str(y))

def test_sym_expr_eq_1():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = 1
    expr2 = 2

    assert(not sr.sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_2():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = 1
    expr2 = d

    assert(sr.sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_3():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')
    
    expr1 = b + 1
    expr2 = d - 2

    assert(sr.sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_4():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = a*x + b
    expr2 = c*x + d

    assert(sr.sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_5():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = a*x + b
    expr2 = c*y + d

    assert(not sr.sym_expr_eq(expr1, expr2, [x, y]))

def test_sym_expr_eq_6():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = (1+a)*x + b + 3
    expr2 = (2+c)*x + d

    assert(sr.sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_7():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = x + a*x + b + 3
    expr2 = 2*x + c*x + d - 1

    assert(sr.sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_8():
    a, b, c, d, e, f, x, y = sympy.symbols('a b c d e f x y')

    expr1 = a*x**2 + 3*x + c
    expr2 = d*x**2 + 2*x + f

    assert(not sr.sym_expr_eq(expr1, expr2, [x]))
"""
def test_sym_expr_eq_9():
    a, b, c, d, e, f, x, y = sympy.symbols('a b c d e f x y')

    expr1 = sympy.sympify("a*log(b*x+c)+d")
    expr2 = sympy.sympify("e+log(g+x*h)*f")

    assert(sym_expr_eq(expr1, expr2, [x]))
"""
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
                  verbose = True,
                  checked_sym_expr = [sympy.sympify("a * x + b * y + c")],
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
                  epsloss = 1e-3)

    model.predict([x, y], np.zeros(len(x)), ["x", "y"])

    assert(len(model.bestExpressions) >= 1)
    assert(sr.sym_expr_eq(model.bestExpressions[0][0], sympy.sympify("(x - x0) ** 2 + (y - y0) ** 2 - R ** 2"), sympy.symbols("x y")))

def test_plane():
    n = 2 * np.random.rand(3) - np.ones(3)
    #n = np.array([1., 2., 3.])
    n /= np.linalg.norm(n)

    u = 2 * np.random.rand(3) - np.ones(3)
    #n = np.array([2., -3., 4.])
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

    assert(len(model.bestExpressions) >= 1)
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

    model.predict(X_train, y_train)

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("x0**2-x1**2+x1-1"))

def test_1():
    #x**2+x+1

    model = sr.SR(niterations = 3,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = x ** 2 + x + 1

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("x ** 2 + x + 1"))
"""
def test_2():
    #sin(x)*exp(x)

    model = sr.SR(niterations = 2,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("d*sin(a*x+b)*exp(c*x+c)+f")],
                  unary_operators = {"sin": (sympy.sin, np.sin),
                                     "exp": (sympy.exp, np.exp)},
                  binary_operators = {"*": (operator.mul, operator.mul)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True,
                  maxfev = 200)

    n = 100
    xmin = -5
    xmax = 10
    x = (xmax - xmin) * np.random.rand(n) + xmin
    y = np.sin(x) * np.exp(x)

    #import matplotlib.pyplot as plt
    #plt.scatter(x, y, label = 'data')
    #x_ = np.linspace(xmin, xmax, 1000)
    #y_ = np.sin(x_) * np.exp(x_)
    #plt.plot(x_, y_, label = 'curve')
    #plt.legend()
    #plt.show()

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("sin(x)*exp(x)"))
"""
def test_3():
    #x/(1+x**2)

    model = sr.SR(niterations = 3,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul),
                                      "/": (operator.truediv, operator.truediv)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True,
                  maxfev = 200)

    n = 10
    xmin = -5
    xmax = 5
    x = (xmax - xmin) * np.random.rand(n) + xmin
    y = x / (1 + x ** 2)

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) >= 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("x / (1 + x **2)"))

def test_4():
    #x**2+y**2

    model = sr.SR(niterations = 3,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = np.random.rand(n)

    model.predict([x, y], x**2 + y **2, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("x**2+y**2"))
"""
def test_5():
    #log(x)+sin(x)

    model = sr.SR(niterations = 2,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("a * log(b * x + c) + d * sin(e * x + f) + g")],
                  unary_operators = {"sin": (sympy.sin, np.sin),
                                     "log": (sympy.log, np.log)},
                  binary_operators = {"+": (operator.add, operator.add)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True)

    n = 10
    xmin = 1
    xmax = 10
    x = (xmax - xmin) * np.random.rand(n) + xmin
    y = np.log(x) + np.sin(x)

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("log(x)+sin(x)"))
"""
"""
def test_6():
    #1/sqrt(2*pi)*exp(-x**2/2)

    model = sr.SR(niterations = 3,
                  unary_operators = {"exp": (sympy.exp, np.exp)},
                  binary_operators = {"*": (operator.mul, operator.mul)},
                  discrete_param_values = [1 / math.sqrt(2 * math.pi), 0, -0.5],
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = np.exp(-x**2 / 2) / math.sqrt(2 * math.pi)

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
"""
def test_koza1():
    #y = x**4 + x**3 + x**2 + x

    model = sr.SR(niterations = 4,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = x**4 + x**3 + x**2 + x

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("x**4 + x**3 + x**2 + x"))

def test_nguyen1():
    #f(x) = x**3 + x**2 + x

    model = sr.SR(niterations = 4,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = x**3 + x**2 + x

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("x**3 + x**2 + x"))

def test_nguyen2():
    #f(x) = x**4 + x**3 + x**2 + x

    model = sr.SR(niterations = 4,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = x**4 + x**3 + x**2 + x

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("x**4 + x**3 + x**2 + x"))

def test_nguyen3():
    #f(x) = x**5 + x**4 + x**3 + x**2 + x

    model = sr.SR(niterations = 4,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = x**5 + x**4 + x**3 + x**2 + x

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("x**5 + x**4 + x**3 + x**2 + x"))

def test_nguyen4():
    #f(x) = x**6 + x**5 + x**4 + x**3 + x**2 + x

    model = sr.SR(niterations = 4,
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = x**6 + x**5 + x**4 + x**3 + x**2 + x

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("x**6 + x**5 + x**4 + x**3 + x**2 + x"))
"""
def test_nguyen5():
    #f(x) = sin(x**2)cos(x)-1

    model = sr.SR(niterations = 3,
                  unary_operators = {"cos": (sympy.cos, np.cos),
                                     "sin": (sympy.sin, np.sin)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = np.sin(x**2)*np.cos(x) - 1

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("sin(x**2)*cos(x)-1"))
"""
"""
def test_nguyen6():
    #f(x) = sin(x)+sin(x+x**2)

    model = sr.SR(niterations = 3,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("f*sin(a*x+b)+g*sin(c*x**2+d*x+e)+h")],
                  unary_operators = {"sin": (sympy.sin, np.sin)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = np.sin(x) + np.sin(x + x**2)

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("sin(x)+sin(x+x**2)"))
"""
"""
def test_nguyen7():
    #f(x) = log(x + 1) + log(x**2 + 1)

    model = sr.SR(niterations = 3,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("f*log(a*x+b)+g*log(c*x**2+d*x+e)+h")],
                  unary_operators = {"log": (sympy.log, np.log)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True,
                  maxfev = 200)

    n = 10
    x = np.random.rand(n)
    y = np.log(x + 1) + np.log(x ** 2 +1)

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("log(x+1)+log(x**2+1)"))
"""
"""
def test_nguyen8():
    #f(x) = sqrt(x)

    model = sr.SR(niterations = 2,
                  unary_operators = {"sqrt": (sympy.sqrt, np.sqrt)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True)

    n = 10
    x = np.random.rand(n)
    y = np.sqrt(x)

    model.predict([x], y, ["x"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0][0] == sympy.sympify("sqrt(x)"))
"""
"""
def test_nguyen9():
    #f(x) = sin(x1)+sin(x2**2)

    model = sr.SR(niterations = 3,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("f*sin(a*x+b)+g*sin(c*x**2+d*x+e)+h")],
                  unary_operators = {"sin": (sympy.sin, np.sin)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True)

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    y = np.sin(x1) + np.sin(x2**2)

    model.predict([x1, x2], y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("sin(x1)+sin(x2**2)"))
"""
"""
def test_nguyen10():
    #f(x) = 2sin(x1)cos(x2)

    model = sr.SR(niterations = 2,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("e*sin(a*x1+b)*cos(c*x2+d)+f")],
                  unary_operators = {"cos": (sympy.cos, np.cos),
                                     "sin": (sympy.sin, np.sin)},
                  binary_operators = {"*": (operator.mul, operator.mul)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True)

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    y = 2 * np.sin(x1) * np.cos(x2)

    model.predict([x1, x2], y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("2*sin(x1)*cos(x2)"))
"""
"""
def test_keijzer10():
    #f(x) = x1**x2

    model = sr.SR(niterations = 2,
                  #verbose = True,
                  #checked_sym_expr = [sympy.sympify("e*(a*x1+b)**(c*x2+d)+f")],
                  binary_operators = {"**": (sympy.Pow, operator.pow)},
                  discrete_param_values = ["(-5, 5)"],
                  foundBreak = True)

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    y = x1 ** x2

    model.predict([x1, x2], y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.sympify("x1**x2"))
"""
