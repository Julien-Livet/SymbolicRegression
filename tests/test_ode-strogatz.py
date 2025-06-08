import math
import matplotlib.pyplot as plt
import numpy as np
import operator
import requests
import sr
import sympy

#https://github.com/lacava/ode-strogatz

loss = math.inf
plot_y = True

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
                  #extra_start_sym_expr = [sympy.sympify("_1235/(_1228*x**2 + _1230*x + _1233) + _1239*x/(_1236*x**2 + _1237*x + _1238) + _1244*y/(_1240*x**2 + _1241*x + _1242) + _1248*x*y/(_1245*x**2 + _1246*x + _1247) + _1251")],
                  discrete_param_values = ["(-2, 2)", 20],
                  operator_depth = {"/": 1, "*": 2, "+": 2},
                  #callback = callback,
                  maxcomplexity = 30,
                  #monothread = True,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.expand(sympy.simplify(sympy.sympify("20 -x - (2*x*y/(2+1*x**2))"))))

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
                  discrete_param_values = ["(-2, 2)", 10],
                  operator_depth = {"/": 1, "*": 2, "+": 2},
                  #callback = callback,
                  maxcomplexity = 30,
                  #monothread = True,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("10 - (x*y/(1+0.5*x**2))")))))

def test_d_barmag1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_barmag1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 0.5*sin(x-y)-sin(x))
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

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("0.5*sin(x-y)-sin(x)")))))

def test_d_barmag2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_barmag2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 0.5*sin(y-x) - sin(y))
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

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("0.5*sin(y-x) - sin(y)")))))

def test_d_glider1():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_glider1.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), -0.05*x**2-sin(y))
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
                  #discrete_param_values = ["(-1, 1)", -0.05, 0.05],
                  operator_depth = {"sin": 1, "*": 2, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("-0.05*x**2-sin(y)")))))

def test_d_glider2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_glider2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), x - cos(y)/x)
    #plt.show()

    global loss, plot_y
    loss = math.inf
    plot_y = True

    plt.ion()

    model = sr.SR(niterations = 3,
                  verbose = False,
                  unary_operators = {"cos": (sympy.cos, np.cos)},
                  binary_operators = {"+": (operator.add, operator.add),
                                      "/": (operator.truediv, operator.truediv)},
                  discrete_param_values = ["(-1, 1)"],
                  operator_depth = {"cos": 1, "/": 1, "+": 1},
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("x - cos(y)/x")))))

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
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("3*x-2*x*y-x**2")))))

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
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("2*y-x*y-y**2")))))

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
                  discrete_param_values = ["(-1, 4)"],
                  operator_depth = {"/": 1, "*": 2, "+": 2},
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

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

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("y*(x/(1+x)-0.075*y)")))))

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
                  #callback = callback,
                  #monothread = True,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("10*(y-(1/3*(x**3-y)))")))))

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

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sr.expr_eq(sympy.expand(model.bestExpressions[0][0]), sympy.expand(sympy.simplify(sympy.sympify("-1/10*x")))))
