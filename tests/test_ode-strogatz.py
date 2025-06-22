import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import requests
import sr
import sympy

#https://github.com/lacava/ode-strogatz

#Commented tests are too long and need some work

loss = math.inf
plot_y = True

def sym_cot(x):
    return sympy.sympify("cot(" + str(x) + ")")

def num_cot(x):
    return 1 / np.tan(x)

def callback(expr, y):
    global loss, plot_y
    
    if (expr.loss < loss):
        print("Best expression", expr.sym_expr, expr.opt_expr, expr.loss)

        loss = expr.loss

        #draw = plt.scatter
        draw = plt.plot

        if (plot_y):
            draw(list(range(0, len(y))), y, label = "label")
            plot_y = False

        f = sympy.lambdify(expr.symbol_vars, expr.opt_expr)
        new_y = f(*expr.value_vars)

        if not (type(new_y) is np.ndarray or type(new_y) is list):
            new_y = [new_y] * len(y)

        draw(list(range(0, len(y))), new_y, label = str(expr.opt_expr))
        plt.legend()
        #plt.show()
        #"""
        plt.draw()
        plt.pause(1.0)
        #"""
    else:
        print("Processed expression", expr.sym_expr, expr.opt_expr, expr.loss)

def file_lines(url):
    response = requests.get(url)
    
    assert(response.status_code == 200)
    
    data = response.text
    
    lines = data.split("\n")

    del lines[0]
    
    if (lines[-1] == ""):
        del lines[-1]

    return lines

def file_data(url):
    lines = file_lines(url)

    label, x, y = [], [], []
    
    for line in lines:
        values = [float(x) for x in line.split(",")]
        label.append(values[0])
        x.append(values[1])
        y.append(values[2])

    label = np.array(label, dtype = np.float64)
    x = np.array(x, dtype = np.float64)
    y = np.array(y, dtype = np.float64)
    
    return label, x, y

def test_d_bacres1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_bacres1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 20 -x - (x*y/(1+0.5*x**2)))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add),
                                      "/": (operator.truediv, operator.truediv)},
                  operator_depth = {"/": 1, "*": 2, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  maxcomplexity = 30,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("20 -x - (*x*y/(1+0.5*x**2))")))))

def test_d_bacres2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_bacres2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 10 - (x*y/(1+0.5*x**2)))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add),
                                      "/": (operator.truediv, operator.truediv)},
                  operator_depth = {"/": 1, "*": 2, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  maxcomplexity = 30,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("10 - (x*y/(1+0.5*x**2))")))))

def test_d_barmag1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_barmag1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 0.5*np.sin(x-y)-np.sin(x))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  unary_operators = {"sin": (sympy.sin, np.sin)},
                  binary_operators = {"+": (operator.add, operator.add)},
                  discrete_param_values = ["(-1, 1)", 0.5],
                  operator_depth = {"sin": 1, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.sympify("0.5*sin(x-y)-sin(x)")))

def test_d_barmag2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_barmag2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 0.5*np.sin(y-x) - np.sin(y))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  unary_operators = {"sin": (sympy.sin, np.sin)},
                  binary_operators = {"+": (operator.add, operator.add)},
                  discrete_param_values = ["(-1, 1)", 0.5],
                  operator_depth = {"sin": 1, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.sympify("0.5*sin(y-x) - sin(y)")))

def test_d_glider1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_glider1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), -0.05*x**2-np.sin(y))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  unary_operators = {"sin": (sympy.sin, np.sin)},
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add)},
                  discrete_param_values = ["(-1, 1)", -0.05],
                  operator_depth = {"sin": 1, "*": 2, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.sympify("-0.05*x**2-sin(y)")))
"""
def test_d_glider2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_glider2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), x - np.cos(y)/x)
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  unary_operators = {"cos": (sympy.cos, np.cos)},
                  binary_operators = {"/": (operator.truediv, operator.truediv),
                                      "+": (operator.add, operator.add)},
                  discrete_param_values = ["(-1, 1)"],
                  operator_depth = {"cos": 1, "/": 1, "+": 1},
                  #callback = callback,
                  #monothread = True,
                  maxcomplexity = 20,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.sympify("x - cos(y)/x")))
"""
def test_d_lv1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_lv1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 3*x-2*x*y-x**2)
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add)},
                  operator_depth = {"*": 3, "+": 3},
                  discrete_param_values = ["(-3, 3)"],
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.sympify("3*x-2*x*y-x**2")))

def test_d_lv2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_lv2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 2*y-x*y-y**2)
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add)},
                  operator_depth = {"*": 3, "+": 3},
                  discrete_param_values = ["(-2, 2)"],
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.sympify("2*y-x*y-y**2")))
"""
def test_d_predprey1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_predprey1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), x*(4-x-y/(1+x)))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add),
                                      "/": (operator.truediv, operator.truediv)},
                  discrete_param_values = ["(-1, 1)", 4],
                  operator_depth = {"/": 1, "*": 2, "+": 4},
                  maxcomplexity = 25,
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("x*(4-x-y/(1+x))")))))

def test_d_predprey2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_predprey2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), y*(x/(1+x)-0.075*y))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add),
                                      "/": (operator.truediv, operator.truediv)},
                  discrete_param_values = ["(-1, 1)", -0.075, 0.075],
                  operator_depth = {"/": 1, "*": 2, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("y*(x/(1+x)-0.075*y)")))))
"""
def test_d_shearflow1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_shearflow1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), num_cot(y)*np.cos(x))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  unary_operators = {"cot": (sym_cot, num_cot),
                                     "cos": (sympy.cos, np.cos)},
                  binary_operators = {"*": (operator.mul, operator.mul)},
                  operator_depth = {"cos": 1, "cot": 1, "*": 2},
                  discrete_param_values = ["(-1, 1)"],
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("cot(y)*cos(x)")))))
"""
def test_d_shearflow2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_shearflow2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), (np.cos(y)**2+0.1*np.sin(y)**2)*np.sin(x))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  unary_operators = {"sin": (sympy.sin, np.sin),
                                     "cos": (sympy.cos, np.cos)},
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add)},
                  operator_depth = {"cos": 1, "sin": 1, "*": 2, "+": 2},
                  discrete_param_values = ["(0, 1)", 0.1],
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("(cos(y)**2+0.1*sin(y)**2)*sin(x)")))))
"""
"""
#This test need some work to find the right expression

def test_d_vdp1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_vdp1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 10*(y-(1/3*(x**3-y))))
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add)},
                  operator_depth = {"*": 2, "+": 2},
                  discrete_param_values = [0, -10/3, 40/3],
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("10*(y-(1/3*(x**3-y)))")))))
"""
def test_d_vdp2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_vdp2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), -1/10*x)
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  binary_operators = {"*": (operator.mul, operator.mul),
                                      "+": (operator.add, operator.add)},
                  operator_depth = {"*": 2, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.fit([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.sympify("-1/10*x")))
