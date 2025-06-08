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
            draw(list(range(0, len(y))), y, label = "y")
            plot_y = False

        f = sympy.lambdify(expr.symbol_vars, expr.opt_expr)
        new_y = f(*expr.value_vars)

        if not (type(new_y) is np.ndarray or type(new_y) is list):
            new_y = [new_y] * len(y)

        draw(list(range(0, len(y))), new_y, label = str(expr.opt_expr))
        plt.legend()
        #plt.show()
        """
        plt.draw()
        plt.pause(1.0)
        """
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
                  #maxfev = 10000,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.expand(sympy.simplify(sympy.sympify("20 -x - (2*x*y/(2+1*x**2))"))))

def test_d_bacres2():
    label, x, y = file_data("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_bacres2.txt")

    #import matplotlib.pyplot as plt
    #plt.scatter(list(range(0, len(label))), label)
    #plt.scatter(list(range(0, len(label))), 10 - (x*y/(1+0.5**x^2)))
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
                  #maxfev = 10000,
                  foundBreak = True)

    model.predict([x, y], label, ["x", "y"])

    assert(len(model.bestExpressions) == 1)
    assert(sympy.expand(model.bestExpressions[0][0]) == sympy.expand(sympy.simplify(sympy.sympify("10 - (2*x*y/(2+1**x^2))"))))
