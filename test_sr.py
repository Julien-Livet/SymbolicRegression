import numpy as np
import operator
import random
import sr
import sympy

def test_x1_mul_x2():
    model = sr.SR(niterations = 3,
                  unary_operators = {"-": (operator.neg, operator.neg)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True,
                  symmetric_binary_operators = ["+", "*"])
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
    #                    "conv": (sympy.Function("conv"), sr.convolve),
    #                    "**": (sympy.Pow, operator.pow}

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = x1 * x2

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0] == sympy.sympify("x1*x2"))

def test_x1_add_x2():
    model = sr.SR(niterations = 3,
                  unary_operators = {"-": (operator.neg, operator.neg)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True,
                  symmetric_binary_operators = ["+", "*"])

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = x1 + x2

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0] == sympy.sympify("x1+x2"))

def test_x1_2_add_x2_2_sub_x1_mul_x2():
    model = sr.SR(niterations = 3,
                  unary_operators = {"-": (operator.neg, operator.neg)},
                  binary_operators = {"*": (operator.mul, operator.mul)},
                  foundBreak = True,
                  symmetric_binary_operators = ["+", "*"])

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = (x1 - x2) ** 2 + x1 * x2

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0]) == sympy.expand(sympy.sympify("(x1-x2)**2+x1*x2")))

def test_a_mul_x1_add_b():
    model = sr.SR(niterations = 3,
                  binary_operators = {"+": (operator.add, operator.add)},
                  foundBreak = True,
                  symmetric_binary_operators = ["+", "*"])

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    a = random.random()
    b = random.random()
    y = a * x1 + b

    model.predict(X, y, ["x1", "x2"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(model.bestExpressions[0], sympy.sympify(str(a) + "*x1+" + str(b))))

def test_a_mul_x2_add_b_mul_x2_add_c():
    model = sr.SR(niterations = 3,
                  binary_operators = {"+": (operator.add, operator.add)},
                  foundBreak = True,
                  symmetric_binary_operators = ["+", "*"])

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
    assert(sr.expr_eq(model.bestExpressions[0], sympy.sympify(str(a) + "*x1+" + str(b) + "*x2+" + str(c))))

def test_pysr():
    X = 2 * np.random.randn(5, 100)
    y = 2.5382 * np.cos(X[3, :]) + X[0, :] ** 2 - 0.5

    model = sr.SR(niterations = 5,
                  unary_operators = {"cos": (sympy.cos, np.cos)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "*": (operator.mul, operator.mul)},
                  foundBreak = True,
                  symmetric_binary_operators = ["+", "*"])

    model.predict(X, y)

    assert(len(model.bestExpressions) == 1)
    assert(model.bestExpressions[0] == sympy.sympify("2.5382 * cos(x3) + x0 ** 2 - 0.5"))
