import copy
from deap import base, creator, tools, algorithms
import itertools
import math
import multiprocessing
import multiprocessing.dummy
import numpy as np
import operator
import pandas as pd
import random
import scipy.optimize
import sympy
import threading

def mse_loss(x, y):
    l = np.sum((x - y) ** 2)

    if (np.isnan(l)):
        return np.inf

    return l

def model_func(func):
    def model(x, *args):
        try:
            return np.array(func(*x, *args), dtype = np.float64)
        except ZeroDivisionError:
            return math.nan

    return model

class Node:
    def __init__(self, symnet, name, op, leaf1, sym_x1 = None, num_x1 = None, leaf2 = None, sym_x2 = None, num_x2 = None):
        self.symnet = symnet
        self.index = symnet.add_param()
        self.name = name
        self.op = op
        self.leaf1 = leaf1
        self.leaf2 = leaf2
        self.sym_x1 = sym_x1
        self.num_x1 = num_x1
        self.sym_x2 = sym_x2
        self.num_x2 = num_x2
        self.considered = True

    def un_ops(self, unary_ops):
        un_ops = set()

        if (self.name in unary_ops):
            un_ops.add(self.name)

        if (type(self.leaf1) is Node):
            un_ops |= self.leaf1.un_ops(unary_ops)
        if (type(self.leaf2) is Node):
            un_ops |= self.leaf2.un_ops(unary_ops)

        return un_ops

    def bin_ops(self, binary_ops):
        bin_ops = set()

        if (self.name in binary_ops):
            bin_ops.add(self.name)

        if (type(self.leaf1) is Node):
            bin_ops |= self.leaf1.bin_ops(binary_ops)
        if (type(self.leaf2) is Node):
            bin_ops |= self.leaf2.bin_ops(binary_ops)

        return bin_ops

    def evalf(self):
        leaf1 = copy.deepcopy(self.leaf1)
        x = 0

        if (isinstance(leaf1, sympy.Symbol)):
            f = sympy.lambdify(self.sym_x1, leaf1, modules = self.symnet.modules)
            x = f(self.num_x1)
        else:
            x = leaf1.evalf()

        leaf2 = copy.deepcopy(self.leaf2)

        if (leaf2 == None):
            if (self.op):
                return self.symnet.a[self.index] * self.op[1](x) + self.symnet.b[self.index]
            else:
                return x

        y = 0

        if (isinstance(leaf2, sympy.Symbol)):
            f = sympy.lambdify(self.sym_x2, leaf2, modules = self.symnet.modules)
            y = f(self.num_x2)
        else:
            y = leaf2.evalf()

        return self.symnet.a[self.index] * self.op[1](x, y) + self.symnet.b[self.index]

    def expr(self):
        a = "a" + str(self.index)
        b = "b" + str(self.index)

        if (isinstance(self.leaf1, sympy.Symbol)):
            return sympy.sympify(a + "*" + str(self.leaf1) + "+" + b)

        if (self.leaf2 == None):
            if (self.op):
                return sympy.sympify(a + "*(" + str(self.op[0](self.leaf1.expr())) + ")+" + b)

        return sympy.sympify(a + "*(" + str(self.op[0](self.leaf1.expr(), self.leaf2.expr())) + ")+" + b)

class SymNet:
    def __init__(self, modules, X, symbols, un_ops, bin_ops, symm_bin_ops):
        self.modules = modules
        self.X = X
        self.symbols = symbols
        self.un_ops = un_ops
        self.bin_ops = bin_ops
        self.un_ops_index = 0
        self.bin_ops_comb = dict(zip(list(self.bin_ops.keys()), [[] for x in self.bin_ops.keys()]))
        self.symm_bin_ops = symm_bin_ops
        self.a = []
        self.b = []
        self.nodes = []

        assert(len(X) == len(symbols))

        for i in range(0, len(X)):
            self.nodes.append(Node(self, "", None, symbols[i], symbols[i], X[i]))

    def add_param(self):
        self.a = np.array(list(self.a) + [0])
        self.b = np.array(list(self.b) + [0])

        return len(self.a) - 1

    def add_layer(self):
        new_nodes = []

        for k, v in self.un_ops.items():
            for n in self.nodes[self.un_ops_index:]:
                if (n.considered):
                    new_nodes.append(Node(self, k, v, n))

        self.un_ops_index = len(self.nodes)
        self.nodes += new_nodes

        new_nodes = []

        for k, v in self.bin_ops.items():
            indices1 = list(range(0, len(self.nodes)))

            for i1 in indices1:
                indices2 = list(range(0, len(self.nodes)))

                for key, value in self.symm_bin_ops.items():
                    if (k == key):
                        indices2 = list(range(i1 + 1 if value else i1, len(self.nodes)))
                        break

                for i2 in indices2:
                    if (not (i1, i2) in self.bin_ops_comb[k]):
                        self.bin_ops_comb[k].append((i1, i2))

                        if (self.nodes[i1].considered and self.nodes[i2].considered):
                            new_nodes.append(Node(self, k, v, self.nodes[i1], leaf2 = self.nodes[i2]))

        self.nodes += new_nodes

    def compute_outputs(self, a, b):
        assert(len(a) == len(self.a))
        assert(len(b) == len(self.b))

        self.a = a
        self.b = b

        outputs = []

        for n in self.nodes:
            outputs.append(n.evalf())

        return outputs

    def expressions(self):
        exprs = []

        for n in self.nodes:
            exprs.append(n.expr())

        return exprs

    def fit(self, y, loss_func = mse_loss, N = 30, bounds = (-10, 10)):
        a_s = []
        b_s = []
        losses = []
        exprs = self.expressions()
        maxfev = 1000
        best_loss = np.inf
        best_a = None
        best_b = None

        for i in range(0, len(exprs)):
            if (not self.nodes[i].considered):
                continue

            expr = exprs[i]

            for j in range(0, N):
                a = np.zeros(len(self.a))
                b = np.zeros(len(self.b))

                ab_syms = list(expr.free_symbols - set(self.symbols))
                f = sympy.lambdify(self.symbols + ab_syms, expr, modules = self.modules)
                func = model_func(f)

                params = np.zeros(len(ab_syms))
                loop = True
                count = 0

                while (loop):
                    params = np.random.uniform(bounds[0], bounds[1], size = len(ab_syms))
                    r = func(self.X, *params)
                    count += 1
                    loop = not np.isfinite(r).all() and count < N

                try:
                    params, _ = scipy.optimize.curve_fit(func, self.X, y, p0 = params, maxfev = maxfev)
                except RuntimeError as e:
                    print(e)

                for i in range(0, len(ab_syms)):
                    index = int(str(ab_syms[i])[1:])

                    if (str(ab_syms[i])[0] == "a"):
                        a[index] = params[i]
                    else:
                        b[index] = params[i]

                l = loss_func(func(self.X, *params), y)

                if (l < best_loss):
                    best_loss = l
                    best_a = a
                    best_b = b

            a_s.append(best_a)
            b_s.append(best_b)
            losses.append(best_loss)

        losses = np.array(losses)
        losses = np.nan_to_num(losses, nan = -np.inf, posinf = -np.inf)
        losses = np.nan_to_num(losses, neginf = np.max(losses))
        losses = 1 - losses / np.max(losses)

        la = sorted(zip(losses, a_s), key = lambda x: x[0], reverse = True)
        lb = sorted(zip(losses, b_s), key = lambda x: x[0], reverse = True)

        self.a = la[0][1]
        self.b = lb[0][1]

        le = sorted(zip(losses, exprs), key = lambda x: x[0], reverse = True)

        m = 0.75 * le[0][0]
        ln = sorted(zip(losses, self.nodes), key = lambda x: x[0], reverse = True)

        for x, y in ln:
            if (x < m):
                y.considered = False

        un_ops = set()
        bin_ops = set()
        best_exprs = []
        best_nodes = []

        for x in ln[:5]:
            best_nodes.append(x[1])
            best_exprs.append(x[1].expr())
            un_ops |= x[1].un_ops(self.un_ops)
            bin_ops |= x[1].bin_ops(self.bin_ops)

        return un_ops, bin_ops, best_exprs, best_nodes

def num_modules(unary_ops, binary_ops):
    modules = ['numpy']

    for name, op in unary_ops.items():
        sym_op, num_op = op

        #if (type(sym_op) == sympy.Function):
        if (name.isidentifier()):
            modules.append({name: num_op})

    for name, op in binary_ops.items():
        sym_op, num_op = op

        #if (type(sym_op) == sympy.core.function.UndefinedFunction):
        if (name.isidentifier()):
            modules.append({name: num_op})

    return modules

def expression_complexity(expr, weights = None):
    if (weights is None):
        weights = {
            sympy.Add: 1,
            sympy.Mul: 2,
            sympy.Pow: 3,
            sympy.log: 3,
            sympy.exp: 3,
            sympy.sin: 3,
            sympy.cos: 3,
            sympy.tan: 3,
            sympy.sqrt: 3,
            sympy.Symbol: 1,
            sympy.Number: 1,
        }

    def traverse(e):
        n_nodes = 1
        depth = 1
        total_weight = weights.get(type(e), 1)

        if (hasattr(e, 'args') and e.args):
            child_depths = []

            for arg in e.args:
                sub = traverse(arg)
                n_nodes += sub['n_nodes']
                total_weight += sub['total_weight']
                child_depths.append(sub['depth'])

            depth += max(child_depths) if child_depths else 0

        return {'n_nodes': n_nodes, 'depth': depth, 'total_weight': total_weight}

    variables = list(expr.free_symbols)

    result = traverse(expr)
    result.update({
        'n_variables': len(variables),
        'n_symbols': len(str(expr))
    })

    return result

def all_delta_discrete_values(discrete_values):
    all_values = all_values_discrete_values(discrete_values)

    all_delta = {}
    n = len(all_values)

    for i in range(n):
        for j in range(n):
            diff = all_values[i] - all_values[j]
            s = all_delta.get(all_values[j], set())
            s.add(diff)
            all_delta[all_values[j]] = s

    return all_delta

def all_values_discrete_values(discrete_values):
    all_values = []

    for value in discrete_values:
        if (type(value) is float):
            all_values.append(value)
        elif (type(value) is int):
            all_values.append(value)
        elif (type(value) == str):
            s = value
            values = [float(x) for x in s[1:-1].split(",")]
            a, b = values[0], values[1]

            assert(a <= b)

            if (s[0] == "(" or s[0] == ")"):
                a, b = int(a), int(b)

                c = 1

                if (len(values) > 2):
                    c = int(values[2])

                all_values += list(range(a, b + 1, c))
            elif (s[0] == "[" or s[0] == "]"):
                c = 10

                if (len(values) > 2):
                    c = int(values[2])

                all_values += list(np.linspace(a, b, c))

    return all_values

def range_discrete_values(discrete_values):
    type_, min_, max_ = int, math.inf, -math.inf

    for value in discrete_values:
        if (type(value) is float):
            type_ = float
            min_ = min(min_, value)
            max_ = max(max_, value)
        elif (type(value) is int):
            type_ = int
            min_ = min(min_, value)
            max_ = max(max_, value)
        elif (type(value) == str):
            s = value
            values = [float(x) for x in s[1:-1].split(",")]
            a, b = values[0], values[1]

            assert(a <= b)

            if (s[0] == "(" or s[0] == ")"):
                a, b = int(a), int(b)
            elif (s[0] == "[" or s[0] == "]"):
                type_ = float

            min_ = min(min_, a)
            max_ = max(max_, b)

    return type_, min_, max_

def random_discrete_values(n, discrete_values):
    values = []

    for i in range(0, n):
        value = discrete_values[random.randint(0, len(discrete_values) - 1)]

        if (type(value) == str):
            s = value
            vv = [float(x) for x in s[1:-1].split(",")]
            a, b = vv[0], vv[1]

            assert(a <= b)

            if (s[0] == "(" or s[0] == ")"):
                a, b = int(a), int(b)
                c = 1

                if (len(vv) > 2):
                    c = int(vv[2])

                if (s[0] == ")"):
                    a += 1

                if (s[-1] == "("):
                    b -=1

                #value = random.randint(a, b)
                value = random.choice(range(a, b, c))
            elif (s[0] == "[" or s[0] == "]"):
                c = 10

                if (len(vv) > 2):
                    c = int(vv[2])
                    value = random.choice(np.linspace(a, b, c))
                else:
                    value = (b - a) * random.random() + a

        values.append(value)

    return values

def round_discrete_values(values, discrete_values):
    for i in range(0, len(values)):
        best_diff = math.inf
        best_value = None

        for v in discrete_values:
            if (type(v) == str):
                s = v
                vv = [float(x) for x in s[1:-1].split(",")]
                a, b = vv[0], vv[1]

                assert(a <= b)

                if (s[0] == "(" or s[0] == ")"):
                    a, b = int(a), int(b)
                    c = 1

                    if (len(vv) > 2):
                        c = int(vv[2])

                    possible_values = range(a, b, c)

                    if (s[0] == ")"):
                        if (possible_values[0] == a):
                            del possible_values[0]
                    if (s[-1] == "("):
                        if (possible_values[-1] == b):
                            del possible_values[-1]
                else:
                    c = 10

                    if (len(vv) > 2):
                        c = int(vv[2])

                    possible_values = np.linspace(a, b, c)

                    if (s[0] == "]"):
                        if (possible_values[0] == a):
                            del possible_values[0]
                    if (s[-1] == "["):
                        if (possible_values[-1] == b):
                            del possible_values[-1]

                if (a <= values[i] and values[i] <= b):
                    if (len(vv) > 2):
                        for x in possible_values:
                            if (abs(values[i] - x) < best_diff):
                                best_diff = abs(values[i] - x)
                                best_value = x
                    else:
                        best_diff = 0
                        best_value = int(round(values[i]))

                    break
                else:
                    if (abs(values[i] - possible_values[0]) < best_diff):
                        best_diff = abs(values[i] - possible_values[0])
                        best_value = possible_values[0]

                    if (abs(values[i] - possible_values[-1]) < best_diff):
                        best_diff = abs(values[i] - possible_values[-1])
                        best_value = possible_values[-1]
            else:
                if (abs(values[i] - v) < best_diff):
                    best_diff = abs(values[i] - v)
                    best_value = v

        values[i] = best_value

    return values

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, elite_size=1):
    bests = []

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fits = list(map(toolbox.evaluate, offspring))

        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit

        elite = tools.selBest(population, elite_size)
        population = tools.selBest(offspring + elite, len(population))
        bests.append(tools.selBest(population, 1)[0])

    return population, bests

def fit(sym_expr, symbol_vars, symbol_params, modules, value_vars, y, p0, loss_func, eps, epsloss, maxfev, discrete_values = [], brute_force_limit = 5e6):
    f = sympy.lambdify(symbol_vars + symbol_params, sym_expr, modules = modules)
    func = model_func(f)

    if (len(discrete_values) == 0):
        value_params = p0
        loop = True
        count = 0

        while (loop):
            value_params = np.random.uniform(-100, 100, size = len(p0))
            r = func(value_vars, *value_params)
            count += 1
            loop = not np.isfinite(r).all() and count < 30

        try:
            value_params, _ = scipy.optimize.curve_fit(func, value_vars, y, p0 = value_params, maxfev = maxfev)
        except RuntimeError as e:
            print(e)

            return p0

        return value_params

    type_, min_, max_ = range_discrete_values(discrete_values)
    all_values = all_values_discrete_values(discrete_values)

    #Step #1: exhaustive algorithm
    if (len(all_values) ** len(p0) < brute_force_limit):
        grid = list(itertools.product(all_values, repeat = len(p0)))

        best_loss = math.inf
        value_params = None

        for params in grid:
            l = loss_func(func(value_vars, *params), y)

            if (l < best_loss):
                best_loss = l
                value_params = params

                if (best_loss < epsloss):
                    break

        return np.array(value_params)

    value_params = random_discrete_values(len(p0), discrete_values)

    #Step #2: remove additive constants and consider (-1, 0, 1) space for exhaustive algorithm
    sym_expr_wo_csts, csts = remove_add_cst_sym_expr(sym_expr, symbol_vars, 1)
    symbol_params_wo_csts = list(set(symbol_params) - set(csts))

    f = sympy.lambdify(symbol_vars + symbol_params_wo_csts, sym_expr_wo_csts, modules = modules)
    func = model_func(f)
    n = len(symbol_params_wo_csts)

    #Search in range (-1, 0, 1) without additive constants
    if (len((-1, 0, 1)) ** n < brute_force_limit):
        grid = list(itertools.product((-1, 0, 1), repeat = n))

        best_loss = math.inf
        value_params = np.zeros(len(symbol_params_wo_csts))

        for params in grid:
            l = loss_func(func(value_vars, *params), y)

            if (l < best_loss):
                best_loss = l
                value_params = params

        p1 = np.array(value_params)

        #Do another exploration by removing null parameters
        pos_values = [0]
        neg_values = [0]

        for v in all_values:
            if (v > 0):
                pos_values.append(v)
            elif (v < 0):
                neg_values.append(v)

        sym_expr_wo_csts, csts = remove_add_cst_sym_expr(sym_expr, symbol_vars, 0)
        se_wo_csts = copy.deepcopy(sym_expr_wo_csts)
        sp_wo_csts = copy.deepcopy(symbol_params_wo_csts)
        params_value = []
        removed_sym = set()

        for i in range(0, len(symbol_params_wo_csts)):
            if (p1[i] == 0):
                se_wo_csts = se_wo_csts.subs(symbol_params_wo_csts[i], 0)
                del sp_wo_csts[sp_wo_csts.index(symbol_params_wo_csts[i])]
                removed_sym.add(symbol_params_wo_csts[i])
            elif (p1[i] > 0):
                params_value.append(pos_values)
            else:#elif (p1[i] < 0):
                params_value.append(neg_values)

        se_wo_csts = se_wo_csts.subs(sympy.zoo, 0)

        for i in reversed(range(0, len(sp_wo_csts))):
            if (not sp_wo_csts[i] in se_wo_csts.free_symbols):
                p1[symbol_params_wo_csts.index(sp_wo_csts[i])] = 0
                removed_sym.add(sp_wo_csts[i])
                del params_value[i]
                del sp_wo_csts[i]

        se_w_csts = sym_expr.subs(dict(zip(removed_sym, [0] * len(removed_sym))))

        for i in reversed(range(0, len(symbol_params))):
            if (not symbol_params[i] in se_w_csts.free_symbols):
                removed_sym.add(symbol_params[i])

        sp = list(set(symbol_params) - removed_sym)

        f = sympy.lambdify(symbol_vars + sp, se_w_csts, modules = modules)
        func = model_func(f)
        n = len(sp)

        #Search in range (-1, 0, 1) with additive constants
        if (len((-1, 0, 1)) ** n >= brute_force_limit):
            value_params = random_discrete_values(len(p0), discrete_values)
        else:
            grid = list(itertools.product((-1, 0, 1), repeat = n))

            best_loss = math.inf
            value_params = np.zeros(len(sp))

            for params in grid:
                l = loss_func(func(value_vars, *params), y)

                if (l < best_loss):
                    best_loss = l
                    value_params = params

            p2 = np.array(value_params)

            se_w_csts = copy.deepcopy(se_w_csts)
            sp_w_csts = copy.deepcopy(sp)
            params_value = []
            removed_sym = set()

            for i in range(0, len(sp)):
                if (p2[i] == 0):
                    se_w_csts = se_w_csts.subs(sp[i], 0)
                    del sp_w_csts[sp_w_csts.index(sp[i])]
                    removed_sym.add(sp[i])
                elif (p2[i] > 0):
                    params_value.append(pos_values)
                else:#elif (p2[i] < 0):
                    params_value.append(neg_values)

            se_w_csts = se_w_csts.subs(sympy.zoo, 0)

            for i in reversed(range(0, len(sp_w_csts))):
                if (not sp_w_csts[i] in se_w_csts.free_symbols):
                    p2[sp.index(sp_w_csts[i])] = 0
                    removed_sym.add(sp_w_csts[i])
                    del params_value[i]
                    del sp_w_csts[i]

            for i in range(0, len(symbol_params)):
                if (not symbol_params[i] in se_w_csts.free_symbols):
                    removed_sym.add(symbol_params[i])

            se_w_csts = copy.deepcopy(sym_expr)

            for x in removed_sym:
                se_w_csts = se_w_csts.subs(x, 0)
                se_w_csts = se_w_csts.subs(sympy.zoo, 0)

            sp = list(set(symbol_params) - removed_sym)

            f = sympy.lambdify(symbol_vars + sp, se_w_csts, modules = modules)
            func = model_func(f)

            grid = list(itertools.product(*params_value))

            #Search in negative or positive ranges with additive constants
            if (len(grid) >= brute_force_limit):
                value_params = random_discrete_values(len(p0), discrete_values)
            else:
                best_loss = math.inf
                value_params = np.zeros(len(sp))

                for params in grid:
                    l = loss_func(func(value_vars, *params), y)

                    if (l < best_loss):
                        best_loss = l
                        value_params = params

                        if (best_loss < epsloss):
                            break

                p3 = np.array(value_params)

                for i in range(0, len(sp)):
                    p2[sp_w_csts.index(sp[i])] = p3[i]

                remplacements = dict(zip(sp_w_csts, p2)) | dict(zip(removed_sym, [0] * len(removed_sym)))
                value_params = [remplacements[val] for i, val in enumerate(symbol_params)]

                final_expr = copy.deepcopy(sym_expr)

                for k, v in dict(zip(symbol_params, value_params)).items():
                    final_expr = final_expr.subs(k, v)
                    final_expr = final_expr.subs(sympy.zoo, 0)

                if (best_loss < epsloss):
                    return value_params

    f = sympy.lambdify(symbol_vars + symbol_params, sym_expr, modules = modules)
    func = model_func(f)
    possible_params = [value_params]

    #Fall back to curve_fit method

    best_loss = np.inf
    params = None

    for j in range(0, 30):
        if (params is None):
            params = value_params
        else:
            loop = True
            count = 0

            while (loop):
                params = random_discrete_values(len(p0), discrete_values)
                r = func(value_vars, *params)
                count += 1
                loop = not np.isfinite(r).all() and count < 30

        try:
            params, _ = scipy.optimize.curve_fit(func, value_vars, y, p0 = params, maxfev = maxfev)
            params = round_discrete_values(params, discrete_values)
        except RuntimeError as e:
            print(e)

        l = loss_func(func(value_vars, *params), y)

        if (l < best_loss):
            best_loss = l
            value_params = params
            possible_params.append(value_params)

            if (l < epsloss):
                return value_params

    #return value_params

    #Fall back to genetic algorithm

    creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
    creator.create("Individual", list, fitness = creator.FitnessMin)

    def evaluate(individual):
        return loss_func(func(value_vars, *individual), y),

    def int_random_param():
        return [random.randint(l, u) for l, u in zip([min_] * len(p0), [max_] * len(p0))]

    def float_random_param():
        return [random.uniform(l, u) for l, u in zip([min_] * len(p0), [max_] * len(p0))]

    toolbox = base.Toolbox()

    if (type_ == int):
        toolbox.register("individual", tools.initIterate, creator.Individual, int_random_param)
        toolbox.register("mutate", tools.mutUniformInt, low = [min_] * len(p0), up = [max_] * len(p0), indpb = 0.2)
    else:
        toolbox.register("individual", tools.initIterate, creator.Individual, float_random_param)
        toolbox.register("mutate", tools.mutPolynomialBounded, eta = 1, low = [min_] * len(p0), up = [max_] * len(p0), indpb = 0.2)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize = 3)

    pop = toolbox.population(n = maxfev)

    for v in possible_params:
        pop.append(creator.Individual(v))

    pop, bests = eaSimpleWithElitism(pop, toolbox, cxpb = 0.5, mutpb = 0.2, ngen = 100)

    value_params = tools.selBest(pop, 1)[0]

    value_params = round_discrete_values(value_params, discrete_values)

    return value_params

def split_list(lst, n):
    k, m = divmod(len(lst), n)

    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def round_val(x, eps):
    r = round(x / eps) * eps

    if (abs(int(r) - r) <= eps):
        return int(r)
    else:
        return r

def remove_add_cst_sym_expr(expr, symbols, val = 0):
    e = expr_terms(expr, symbols)
    e = sorted(e, key = str)
    expr = sympy.collect(expr, e)
    csts = []

    for arg in expr.args:
        e, cst = remove_add_cst_sym_expr(arg, symbols, val)
        expr = expr.subs(arg, e)
        csts += cst

    expr = sympy.simplify(expr)

    if (expr.is_Add):
        nums = []
        ws_exprs = []
        wos_exprs = []

        for arg in expr.args:
            if arg.is_Number:
                nums.append(arg)
            else:
                found = False

                for s in symbols:
                    if (arg.has(s)):
                        found = True
                        break

                if (found):
                    ws_exprs.append(arg)
                else:
                    wos_exprs.append(arg)

        if (len(wos_exprs)):
            for n in nums:
                expr = expr.subs(n, 0)

            for e in wos_exprs[1:]:
                expr = expr.subs(e, 0)

            expr = expr.subs(wos_exprs[0], val)
            csts.append(wos_exprs[0])

            expr = sympy.simplify(expr)

    return expr, csts

def symplify_sym_expr(expr, symbols):
    e = expr_terms(expr, symbols)
    e = sorted(e, key = str)
    expr = sympy.collect(expr, e)

    for arg in expr.args:
        expr = expr.subs(arg, symplify_sym_expr(arg, symbols))

    expr = sympy.simplify(expr)

    if (expr.is_Add):
        nums = []
        ws_exprs = []
        wos_exprs = []

        for arg in expr.args:
            if arg.is_Number:
                nums.append(arg)
            else:
                found = False

                for s in symbols:
                    if (arg.has(s)):
                        found = True
                        break

                if (found):
                    ws_exprs.append(arg)
                else:
                    wos_exprs.append(arg)

        if (len(wos_exprs)):
            for n in nums:
                expr = expr.subs(n, 0)

            for e in wos_exprs[1:]:
                expr = expr.subs(e, 0)

            expr = expr.subs(wos_exprs[0], newSymbol())

            expr = sympy.simplify(expr)
    elif (expr.is_Mul):
        nums = []
        ws_exprs = []
        wos_exprs = []

        for arg in expr.args:
            if arg.is_Number:
                nums.append(arg)
            else:
                found = False

                for s in symbols:
                    if (arg.has(s)):
                        found = True
                        break

                if (found):
                    ws_exprs.append(arg)
                else:
                    wos_exprs.append(arg)

        if (len(wos_exprs)):
            for n in nums:
                expr = expr.subs(n, 1)

            for e in wos_exprs[1:]:
                expr = expr.subs(e, 1)

            expr = expr.subs(wos_exprs[0], newSymbol())

            expr = sympy.simplify(expr)

    return expr

def same_ast_structure(expr1, expr2, symbols):
    if type(expr1) != type(expr2):
        if not ((expr1.is_Number and expr2.is_Symbol)
                or (expr2.is_Number and expr1.is_Symbol)):
            return False
    else:
        if (expr1.is_Symbol):
            if (expr1 in symbols or expr2 in symbols):
                if (expr1 != expr2):
                    return False
        elif (expr1.is_number):
            if (expr1 != expr2):
                return False

    if len(expr1.args) != len(expr2.args):
        return False

    args1 = expr1.args

    if (len(args1)):
        e = [expr_terms(arg, symbols) for arg in args1]
        e, args1 = list(zip(*sorted(zip(e, args1), key = lambda x: str(x[0]))))

    args2 = expr2.args

    if (len(args2)):
        e = [expr_terms(arg, symbols) for arg in args2]
        e, args2 = list(zip(*sorted(zip(e, args2), key = lambda x: str(x[0]))))

    return all(same_ast_structure(a1, a2, symbols) for a1, a2 in zip(args1, args2))

def expr_terms(expr, symbols):
    terms = []

    if (expr.is_Add):
        return [expr_terms(a, symbols)[0] for a in expr.args]
    elif (expr.is_Mul):
        numbers = []
        syms = []
        has_expr = False

        for arg in expr.args:
            if arg.is_Number:
                numbers.append(arg)
            elif arg.is_Symbol:
                syms.append(arg)
            else:
                has_expr = True

        e = expr
        l = len(syms)
        syms = list(set(syms) - set(symbols))

        if (has_expr):
            for n in numbers:
                e = e.subs(n, 1)

            for s in syms:
                e = e.subs(s, 1)

            e = sympy.simplify(e)
        else:
            if (l):
                for n in numbers:
                    e = e.subs(n, 1)

                for s in syms:
                    e = e.subs(s, 1)

                e = sympy.simplify(e)

        terms = [e]
    else:
        terms.append(expr)

    return terms

def sym_expr_eq(a, b, symbols = []):
    a = sympy.expand(sympy.sympify(a))
    a = symplify_sym_expr(a, symbols)
    a = sympy.expand(a)
    b = sympy.expand(sympy.sympify(b))
    b = symplify_sym_expr(b, symbols)
    b = sympy.expand(b)

    return same_ast_structure(a, b, symbols)

def expr_eq(expr1, expr2, subs_expr = {}, eps = 1e-3):
    expr = sympy.factor(sympy.sympify(expr1 - expr2))

    for key, value in subs_expr.items():
        q, r = sympy.div(expr, key)
        expr = (value * q + r).subs(key, value)

    expr = sympy.simplify(expr)

    s = str(expr)

    if ("cos" in s or "sin" in s or "tan" in s or "sec" in s):
        expr = sympy.trigsimp(sympy.expand_trig(expr))

    expr = expr.replace(
        lambda e: e.is_Number,
        lambda e: round_val(e.evalf(), eps)
    )

    return sympy.simplify(expr) == sympy.sympify("0")

def sign(num):
    return -1 if num < 0 else 1

symbolIndex = 0

def init_shared(var_):
    global shared_symbolIndex
    shared_symbolIndex = var_

def newSymbol():
    try:
        i = shared_symbolIndex.value
        shared_symbolIndex.value += 1
    except NameError:
        global symbolIndex

        i = symbolIndex
        symbolIndex += 1

    return sympy.Symbol("_" + str(i))

def new_params(expr, symbols):
    e = symplify_sym_expr(expr, symbols)

    new_symbol_params = list(e.free_symbols - set(symbols))
    new_value_params = np.ones(len(new_symbol_params))

    return (new_symbol_params, new_value_params, e)

class Expr:
    id_count = multiprocessing.Value('i', 0)
    _lock = threading.Lock()

    def __init__(self, symbol_var = None, value_var = None, expr = None, symbol_vars = None, value_vars = None):
        self.op_tree = []

        with Expr._lock:
            self.id = Expr.id_count.value
            Expr.id_count.value += 1

        if (expr and symbol_vars and value_vars):
            self.symbol_vars = []
            self.value_vars = []

            for s in expr.free_symbols:
                try:
                    i = symbol_vars.index(s)
                    self.symbol_vars.append(symbol_vars[i])
                    self.value_vars.append(value_vars[i])
                except:
                    pass

            self.symbol_params = list(expr.free_symbols - set(symbol_vars))
            self.value_params = np.ones(len(self.symbol_params))

            assert(len(self.symbol_params) == len(self.value_params))

            self.sym_expr = expr

            self.simplify()
        else:
            a = newSymbol()
            b = newSymbol()

            self.sym_expr = a * symbol_var + b
            self.symbol_vars = [symbol_var]
            self.value_vars = [value_var]
            self.symbol_params = [a, b]
            self.value_params = np.array([1.0, 0.0])

            assert(len(self.symbol_params) == len(self.value_params))

        self.opt_expr = ""
        self.loss = math.inf

    def compute_opt_expr(self, y, loss_func, subs_expr, eps, unary_ops, binary_ops, maxfev, epsloss, fixed_cst_value = None, discrete_param_values = [], brute_force_limit = 5e6):
        modules = num_modules(unary_ops, binary_ops)

        symbol_params = copy.deepcopy(self.symbol_params)
        value_params = list(copy.deepcopy(self.value_params))
        sym_expr = copy.deepcopy(self.sym_expr)

        if (fixed_cst_value != None):
            sym_expr = sym_expr.subs(symbol_params[-1], fixed_cst_value)
            del symbol_params[-1]
            del value_params[-1]

        try:
            p0 = [float(x) for x in value_params]
            #p0 = np.random.randn(len(p0), 1)

            if (len(p0) <= len(y)):
                value_params = fit(sym_expr, self.symbol_vars, symbol_params, modules, self.value_vars, y, p0, loss_func, eps, epsloss, maxfev, discrete_param_values, brute_force_limit)

            for i in range(0, len(value_params)):
                value_params[i] = round(value_params[i] / eps) * eps

                if (abs(self.value_params[i]) < eps):
                    value_params[i] = 0

            if (fixed_cst_value != None):
                self.value_params = list(value_params) + [fixed_cst_value]
            else:
                self.value_params = value_params

            self.value_params = np.array(self.value_params)

            norm = np.linalg.norm(self.value_params)

            if (norm > eps):
                p = self.value_params / norm

                for i in range(0, len(p)):
                    if (abs(abs(p[i]) - 1) < eps):
                        p[i] = sign(p[i])

                self.value_params = norm * p
            else:
                self.value_params *= 0
        except RuntimeError:
            pass

        self.opt_expr = copy.deepcopy(sym_expr)

        for i in range(0, len(symbol_params)):
            v = int(self.value_params[i]) if self.value_params[i] == int(self.value_params[i]) else self.value_params[i]
            self.opt_expr = self.opt_expr.subs(symbol_params[i], v)
            self.opt_expr = self.opt_expr.subs(sympy.zoo, 0)

        self.opt_expr = sympy.simplify(self.opt_expr)

        f = sympy.lambdify(self.symbol_vars, self.opt_expr, modules = modules)
        y_pred = f(*self.value_vars)
        self.loss = loss_func(y_pred, y)

        if (self.loss < epsloss):
            self.opt_expr = sympy.factor(sympy.sympify(self.opt_expr))

            for key, value in subs_expr.items():
                q, r = sympy.div(self.opt_expr, key)
                self.opt_expr = sympy.simplify((value * q + r).subs(key, value))

            s = str(self.opt_expr)

            if ("cos" in s or "sin" in s or "tan" in s or "sec" in s):
                self.opt_expr = sympy.trigsimp(sympy.expand_trig(self.opt_expr))

            self.opt_expr = self.opt_expr.replace(
                lambda e: e.is_Number,
                lambda e: round_val(e.evalf(), eps)
            )

    def apply_unary_op(self, unary_sym_num_op):
        expr = copy.deepcopy(self)

        with Expr._lock:
            expr.id = Expr.id_count.value
            Expr.id_count.value += 1
        
        sym_op, num_op = unary_sym_num_op

        a = newSymbol()
        while (a in expr.symbol_params):
            a = newSymbol()
        b = newSymbol()
        while (b in expr.symbol_params):
            b = newSymbol()

        expr.sym_expr = a * sym_op(expr.sym_expr) + b
        expr.symbol_params = list(expr.symbol_params) + [a, b]
        expr.value_params = list(expr.value_params) + [1.0, 0.0]

        assert(len(expr.symbol_params) == len(expr.value_params))
        assert(len(expr.symbol_params) == len(set(expr.symbol_params)))

        expr.simplify()
        expr.op_tree.append(sym_op)

        return expr

    def apply_binary_op(self, binary_sym_num_op, other_expr):
        s1 = copy.deepcopy(self.symbol_params)
        s2 = copy.deepcopy(other_expr.symbol_params)

        expr = copy.deepcopy(self)

        with Expr._lock:
            expr.id = Expr.id_count.value
            Expr.id_count.value += 1

        sym_op, num_op = binary_sym_num_op

        symbol_params = []
        other_sym_expr = other_expr.sym_expr

        for i in range(0, len(other_expr.symbol_params)):
            s = newSymbol()
            while (s in symbol_params or s in s1):
                s = newSymbol()
            symbol_params.append(s)
            other_sym_expr = other_sym_expr.subs(other_expr.symbol_params[i], symbol_params[-1])

        a = newSymbol()
        while (a in symbol_params or a in s1):
            a = newSymbol()
        b = newSymbol()
        while (b in symbol_params or b in s1):
            b = newSymbol()

        expr.sym_expr = a * sym_op(expr.sym_expr, other_sym_expr) + b

        expr.symbol_vars += other_expr.symbol_vars
        expr.value_vars += other_expr.value_vars

        seen = set()
        unique_symbols = []
        unique_values = []

        for sym, val in zip(expr.symbol_vars, expr.value_vars):
            if (sym not in seen):
                seen.add(sym)
                unique_symbols.append(sym)
                unique_values.append(val)

        expr.symbol_vars = unique_symbols
        expr.value_vars = unique_values

        expr.symbol_params += symbol_params + [a, b]
        expr.value_params = list(expr.value_params) + list(other_expr.value_params) + list(np.array([1.0, 0.0]))

        expr.simplify()

        assert(len(expr.symbol_params) == len(expr.value_params))
        assert(len(expr.symbol_params) == len(set(expr.symbol_params)))

        expr.op_tree += other_expr.op_tree
        expr.op_tree.append(sym_op)

        return expr

    def simplify(self):
        sym_expr = sympy.factor(sympy.expand(self.sym_expr))

        n_p = new_params(sym_expr, self.symbol_vars)

        self.symbol_params = n_p[0]
        self.value_params = n_p[1]
        self.sym_expr = n_p[2]

        return self

def eval_binary_combination(args):
    expr1, expr2, name, opt_exps, binary_operator, y, loss_func, maxloss, verbose, eps, epsloss, avoided_expr, foundBreak, subs_expr, un_ops, bin_ops, maxfev, fixed_cst_value, discrete_param_values, groupId, taskId, process_sym_expr, symbols, operator_depth, callback, brute_force_limit, shared_finished = args

    if (shared_finished.value):
        return None

    process = True

    if (process_sym_expr != None):
        process = False

        for e in process_sym_expr:
            if (sym_expr_eq(e, expr1.sym_expr, symbols) or sym_expr_eq(e, expr2.sym_expr, symbols)):
                process = True
                break

    if (not process):
        return None

    if (verbose):
        print("Operator " + name + " group #" + str(groupId) + " task #" + str(taskId))

    depth1 = expr1.op_tree.count(binary_operator[0])
    depth2 = expr2.op_tree.count(binary_operator[0])

    if (depth1 >= operator_depth.get(name, math.inf)
        or depth2 >= operator_depth.get(name, math.inf)):
        return None

    new_expr = expr1.apply_binary_op(binary_operator, expr2)

    try:
        new_expr.compute_opt_expr(y, loss_func, subs_expr, eps, un_ops, bin_ops, maxfev, epsloss, fixed_cst_value, discrete_param_values, brute_force_limit)
    except ZeroDivisionError:
        return None

    #if (verbose):
    #    print("Compute optimal expression (" + str(len(new_expr.sym_expr.free_symbols - set(symbols))) + " parameters): " + str(new_expr.sym_expr), str(new_expr.opt_expr), str(new_expr.loss))

    if (callback):
        callback(new_expr, y)

    if (new_expr.loss <= maxloss):
        if (not new_expr.opt_expr in avoided_expr):
            if (new_expr.loss < epsloss and foundBreak):
                if (verbose):
                    print("Found expression:", str(new_expr.opt_expr), new_expr.loss)

                shared_finished.value = True

        return new_expr

    return None

class SR:
    def __init__(self,
                 niterations = 20,
                 unary_operators = {},
                 binary_operators = {},
                 elementwise_loss = mse_loss,
                 foundBreak = False,
                 maxloss = math.inf,
                 maxexpr = -1,
                 discard_previous_expr = False,
                 symmetric_binary_operators = {"+": True, "-": True, "*": False, "conv": False, "fmin": True, "fmax": True}, #True for strict symmetry
                 shuffle_indices = False,
                 verbose = False,
                 group_expr_size = -1,
                 eps = 1e-3,
                 epsloss = 1e-9,
                 avoided_expr = [],
                 subs_expr = {},
                 sort_by_loss = False,
                 maxfev = 1000,
                 checked_sym_expr = [],
                 extra_start_sym_expr = [],
                 fixed_cst_value = None,
                 maxtask = -1,
                 discrete_param_values = [],
                 process_sym_expr = None,
                 op_weights = None,
                 operator_depth = {},
                 callback = None,
                 maxcomplexity = -1,
                 monothread = False,
                 brute_force_limit = 5e6,
                 auto_ops = False,
                 csv_filename = None,
                 auto_ops_depth = (2, 1),
                 ):
        self.niterations = niterations
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.elementwise_loss = elementwise_loss
        self.foundBreak = foundBreak
        self.maxloss = maxloss
        self.maxexpr = maxexpr
        self.discard_previous_expr = discard_previous_expr
        self.symmetric_binary_operators = symmetric_binary_operators
        self.shuffle_indices = shuffle_indices
        self.verbose = verbose
        self.group_expr_size = group_expr_size
        self.eps = eps
        self.epsloss = epsloss
        self.avoided_expr = avoided_expr
        self.subs_expr = subs_expr
        self.sort_by_loss = sort_by_loss
        self.maxfev = maxfev
        self.checked_sym_expr = checked_sym_expr
        self.extra_start_sym_expr = extra_start_sym_expr
        self.fixed_cst_value = fixed_cst_value
        self.maxtask = maxtask
        self.discrete_param_values = discrete_param_values
        self.process_sym_expr = process_sym_expr
        self.op_weights = op_weights
        self.operator_depth = operator_depth
        self.callback = callback
        self.maxcomplexity = maxcomplexity
        self.monothread = monothread
        self.brute_force_limit = brute_force_limit
        self.auto_ops = auto_ops
        self.csv_filename = csv_filename
        self.auto_ops_depth = auto_ops_depth

        assert(self.eps > 0)

        self.expressions = []
        self.bestExpressions = []
        self.lastIteration = -1

        if (self.auto_ops):
            if (len(self.unary_operators) == 0):
                self.unary_operators = {"neg": (sym_neg, operator.neg),
                                        "abs": (sympy.Abs, operator.abs),
                                        "inv_": (sym_inv_, num_inv_),
                                        #"sqrt": (sym_sqrt, np.sqrt), #not useful because exp and log combination
                                        "cos": (sympy.cos, np.cos),
                                        #"sin": (sympy.sin, np.sin), #not useful, same as cos with phase
                                        "tan": (sympy.tan, np.tan),
                                        #"acos": (sympy.acos, np.arccos),
                                        #"asin": (sympy.asin, np.arcsin),
                                        #"atan": (sympy.atan, np.arctan),
                                        "log": (sympy.log, np.log),
                                        "exp": (sympy.exp, np.exp),
                                        #"sinh": (sympy.sinh, np.sinh),
                                        #"cosh": (sympy.cosh, np.cosh),
                                        #"tanh": (sympy.tanh, np.tanh),
                                        #"asinh": (sympy.asinh, np.arcsinh),
                                        #"acosh": (sympy.acosh, np.arccosh),
                                        #"atanh": (sympy.atanh, np.arctanh),
                                        "floor": (sym_floor, np.floor),
                                        "ceil": (sym_ceil, np.ceil)}

            if (len(self.binary_operators) == 0):
                self.binary_operators = {"+": (operator.add, operator.add),
                                         #"-": (operator.sub, operator.sub), #equivalent to "+"
                                         "*": (operator.mul, operator.mul),
                                         "/": (operator.truediv, operator.truediv),
                                         #"//": (operator.floordiv, operator.floordiv),
                                         #"%": (operator.mod, operator.mod),
                                         #"**": (sympy.Pow, operator.pow),
                                         #"conv": (sym_conv, convolve),
                                         #"fmin": (sym_fmin, np.fmin),
                                         #"fmax": (sym_fmax, np.fmax),
                                         }

    def fit(self, X, y, variable_names = []):
        y = np.array(y, dtype = np.float64)

        self.lastIteration = -1
        self.expressions = []

        given_symbols = sympy.symbols(" ".join(variable_names)) if len(variable_names) else []

        if (type(given_symbols) == sympy.Symbol):
            given_symbols = [given_symbols]

        default_symbols = sympy.symbols(" ".join("x" + str(i) for i in range(0, len(X))))

        if (type(default_symbols) == sympy.Symbol):
            default_symbols = [default_symbols]

        symbols = given_symbols

        if (len(symbols) < len(default_symbols)):
            symbols += default_symbols[len(given_symbols) - len(default_symbols):]
        elif (len(symbols) > len(X)):
            symbols = symbols[:len(X)]

        unary_operators = copy.deepcopy(self.unary_operators)
        binary_operators = copy.deepcopy(self.binary_operators)
        extra_start_sym_expr = copy.deepcopy(self.extra_start_sym_expr)

        if (self.auto_ops):
            modules = num_modules(unary_operators, binary_operators)

            symnet_un = SymNet(modules, X, symbols, unary_operators, {}, self.symmetric_binary_operators)

            for i in range(0, self.auto_ops_depth[0]):
                symnet_un.add_layer()
                un_ops_un, bin_ops_un, best_exprs_un, best_nodes_un = symnet_un.fit(y, self.elementwise_loss)

            if (self.verbose):
                print("Auto unary operators", best_exprs_un, un_ops_un)

            symnet_bin = SymNet(modules, X, symbols, {}, binary_operators, self.symmetric_binary_operators)

            for i in range(0, self.auto_ops_depth[1]):
                symnet_bin.nodes = best_nodes_un
                symnet_bin.add_layer()

            un_ops_bin, bin_ops_bin, best_exprs_bin, best_nodes_bin = symnet_bin.fit(y, self.elementwise_loss)

            if (self.verbose):
                print("Auto binary operators", best_exprs_bin, bin_ops_bin)

            keys = []
            for k, v in unary_operators.items():
                if (not k in un_ops_un):
                    keys.append(k)
            for k in keys:
                del unary_operators[k]

            keys = []
            for k, v in binary_operators.items():
                if (not k in bin_ops_bin):
                    keys.append(k)
            for k in keys:
                del binary_operators[k]

            extra_start_sym_expr += best_exprs_bin

        un_comb = dict(zip(list(unary_operators.keys()), [[] for x in unary_operators.keys()]))
        bin_comb = dict(zip(list(binary_operators.keys()), [[] for x in binary_operators.keys()]))

        exprs = []
        opt_exprs = {}

        for i in range(0, len(symbols)):
            exprs.append(Expr(symbols[i], X[i]))
            exprs[-1].compute_opt_expr(y, self.elementwise_loss, self.subs_expr, self.eps, unary_operators,
                                       binary_operators, self.maxfev, self.epsloss, self.fixed_cst_value,
                                       self.discrete_param_values, self.brute_force_limit)
            opt_exprs[str(exprs[-1].opt_expr)] = exprs[-1].loss

            if (self.callback):
                self.callback(exprs[-1], y)

        for ee in extra_start_sym_expr:
            exprs.append(Expr(expr = ee, symbol_vars = symbols, value_vars = X))
            exprs[-1].compute_opt_expr(y, self.elementwise_loss, self.subs_expr, self.eps, unary_operators,
                                       binary_operators, self.maxfev, self.epsloss, self.fixed_cst_value,
                                       self.discrete_param_values, self.brute_force_limit)
            opt_exprs[str(exprs[-1].opt_expr)] = exprs[-1].loss
            
            if (self.verbose):
                print("Extra start symbolic expression", exprs[-1].sym_expr, exprs[-1].opt_expr, exprs[-1].loss)

            if (self.callback):
                self.callback(exprs[-1], y)

        if (self.verbose):
            print("Best expression", min(opt_exprs, key = opt_exprs.get), min(opt_exprs.values()))

        self.expressions = opt_exprs

        losses = [value for key, value in opt_exprs.items()]
        sortedLosses, sortedOpt_exprs = zip(*sorted(zip(losses, list(opt_exprs.keys()))))

        self.bestExpressions = []

        for i in range(0, len(sortedLosses)):
            if (sortedLosses[i] >= self.epsloss):
                break

            expr = sympy.simplify(sortedOpt_exprs[i])

            if (not expr in self.avoided_expr):
                self.bestExpressions.append((expr, sortedLosses[i]))

        if (len(self.bestExpressions)):
            if (self.csv_filename):
                df = pd.DataFrame()

                df_sym_expr = []
                df_opt_expr = []
                df_loss = []
                df_sym_complexity = []
                df_opt_complexity = []

                for e in self.expressions:
                    df_sym_expr.append(e.sym_expr)
                    df_opt_expr.append(e.opt_expr)
                    df_loss.append(e.loss)
                    df_sym_complexity.append(expression_complexity(e.sym_expr, self.op_weights)["total_weight"])
                    df_opt_complexity.append(expression_complexity(e.opt_expr, self.op_weights)["total_weight"])

                df["sym_expr"] = df_sym_expr
                df["opt_expr"] = df_opt_expr
                df["loss"] = df_loss
                df["sym_complexity"] = df_sym_complexity
                df["opt_complexity"] = df_opt_complexity

                df = df.sort_values(by = "loss")

                df.to_csv(self.csv_filename)

            return

        self.expressions = exprs

        for j in range(0, self.niterations):
            if (self.verbose):
                print("Iteration #" + str(j))

            self.lastIteration = j

            finished = False

            newExprs = []

            exprs_to_process = []

            for name, unary_operator in unary_operators.items():
                for expr in exprs:
                    if (expr.id in un_comb[name]):
                        continue

                    un_comb[name].append(expr.id)

                    process = True

                    if (self.process_sym_expr != None):
                        process = False

                        for e in self.process_sym_expr:
                            if (sym_expr_eq(e, expr.sym_expr, symbols)):
                                process = True
                                break

                    if (process):
                        exprs_to_process.append((expr, unary_operator, name))

            exprs_to_process = sorted(exprs_to_process, key = lambda x: expression_complexity(x[0].sym_expr, self.op_weights)["total_weight"])

            for expr in exprs_to_process:
                depth = expr[0].op_tree.count(expr[1][0])

                if (depth < self.operator_depth.get(expr[2], math.inf)):
                    new_expr = expr[0].apply_unary_op(expr[1])

                    try:
                        new_expr.compute_opt_expr(y, self.elementwise_loss, self.subs_expr, self.eps, unary_operators,
                                                  binary_operators, self.maxfev, self.epsloss, self.fixed_cst_value,
                                                  self.discrete_param_values, self.brute_force_limit)

                        if (self.callback):
                            self.callback(new_expr, y)

                        if (new_expr.loss <= self.maxloss):
                            if (not new_expr.opt_expr in self.avoided_expr):
                                newExprs.append(new_expr)
                                opt_exprs[str(new_expr.opt_expr)] = new_expr.loss

                                if (new_expr.loss < self.epsloss):
                                    if (self.verbose):
                                        print("Found expression:", str(new_expr.opt_expr), new_expr.loss)

                                    if (self.foundBreak):
                                        finished = True
                                        break
                    except ZeroDivisionError:
                        pass

                    if (finished):
                        break

            if (len(unary_operators)):
                if (self.discard_previous_expr):
                    exprs = newExprs
                else:
                    exprs += newExprs

                if (self.sort_by_loss):
                    exprs = sorted(exprs, key=lambda x: x.loss)

            if (self.verbose):
                print("Best expression", min(opt_exprs, key = opt_exprs.get), min(opt_exprs.values()))

                for k1 in range(0, len(self.checked_sym_expr)):
                    ce = self.checked_sym_expr[k1]

                    for k2 in range(0, len(exprs)):
                        e = exprs[k2]

                        if (sym_expr_eq(e.sym_expr, ce, symbols)):
                            print("Checked expression", ce, k1, k2, e.opt_expr, e.loss)

            newExprs = []

            if (finished):
                break

            tasks = []
            groups = split_list(exprs, len(exprs) // self.group_expr_size + 1) if self.group_expr_size > 0 else [exprs]

            with multiprocessing.Manager() as manager:
                shared_finished = manager.Value('b', False)

                for name, binary_operator in binary_operators.items():
                    for groupId in range(0, len(groups)):
                        group = groups[groupId]
                        indices1 = list(range(0, len(group)))

                        if (self.shuffle_indices):
                            random.shuffle(indices1)

                        n = len(tasks)

                        newTasks = []

                        for i1 in indices1:
                            indices2 = list(range(0, len(group)))

                            for k, v in self.symmetric_binary_operators.items():
                                if (name == k):
                                    indices2 = list(range(i1 + 1 if v else i1, len(group)))
                                    break

                            if (self.shuffle_indices):
                                random.shuffle(indices2)

                            for i2 in indices2:

                                if ((group[i1].id, group[i2].id) in bin_comb[name]):
                                    continue

                                bin_comb[name].append((group[i1].id, group[i2].id))

                                newTasks.append((group[i1], group[i2], name, opt_exprs, binary_operator, y,
                                                self.elementwise_loss, self.maxloss, self.verbose,
                                                self.eps, self.epsloss, self.avoided_expr, self.foundBreak, self.subs_expr,
                                                unary_operators, binary_operators, self.maxfev, self.fixed_cst_value,
                                                self.discrete_param_values, groupId, len(tasks) - n + len(newTasks),
                                                self.process_sym_expr, symbols, self.operator_depth, self.callback,
                                                self.brute_force_limit, shared_finished))

                        if (self.maxtask > 0):
                            newTasks = newTasks[:self.maxtask]

                        tasks += newTasks

                        if (self.verbose):
                            print("Operator " + name + " group #" + str(groupId) + " with " + str(len(tasks) - n) + " tasks")

                tasks = sorted(tasks, key = lambda x: expression_complexity(x[0].sym_expr, self.op_weights)["total_weight"]
                                                      + expression_complexity(x[1].sym_expr, self.op_weights)["total_weight"])

                results = []

                shared_value = manager.Value('i', symbolIndex)

                if (self.monothread):
                    if (self.verbose):
                        print("Monothread process")

                    for t in tasks:
                        results.append(eval_binary_combination(t))
                else:
                    import _pickle

                    try:
                        if (self.verbose):
                            print("multiprocessing process")

                        with multiprocessing.Pool(initializer = init_shared, initargs = (shared_value,), processes = multiprocessing.cpu_count()) as pool:
                            results = pool.map(eval_binary_combination, tasks)
                    except (_pickle.PicklingError, RuntimeError, AttributeError) as e:
                        try:
                            if (self.verbose):
                                print("multiprocessing.dummy process")

                            with multiprocessing.dummy.Pool(initializer = init_shared, initargs = (shared_value,), processes = multiprocessing.cpu_count()) as pool:
                                results = pool.map(eval_binary_combination, tasks)
                        except BrokenPipeError:
                            if (self.verbose):
                                print("Monothread process")

                            for t in tasks:
                                results.append(eval_binary_combination(t))

                finished = shared_finished.value

                for res in results:
                    if (res is not None):
                        newExprs.append(res)
                        opt_exprs[str(res.opt_expr)] = res.loss

            if (self.discard_previous_expr):
                exprs = newExprs
            else:
                exprs += newExprs

            if (self.verbose):
                print("Best expression", min(opt_exprs, key = opt_exprs.get), min(opt_exprs.values()))

                for k1 in range(0, len(self.checked_sym_expr)):
                    ce = self.checked_sym_expr[k1]

                    for k2 in range(0, len(exprs)):
                        e = exprs[k2]

                        if (sym_expr_eq(e.sym_expr, ce, symbols)):
                            print("Checked expression", ce, k1, k2, e.opt_expr, e.loss)

            if (self.maxcomplexity > 0):
                exprs = sorted(exprs, key = lambda x: expression_complexity(x.sym_expr, self.op_weights)["total_weight"])
                i = len(exprs) - 1

                while (i > 0):
                    if (expression_complexity(exprs[i].sym_expr, self.op_weights)["total_weight"] <= self.maxcomplexity):
                        break

                    i -= 1

                exprs = exprs[:i]

            if (self.sort_by_loss):
                exprs = sorted(exprs, key = lambda x: x.loss)

            if (self.maxexpr > 0 and len(exprs) > self.maxexpr):
                exprs = exprs[:self.maxexpr]

            if (finished):
                break

        self.expressions = exprs

        losses = [value for key, value in opt_exprs.items()]
        sortedLosses, sortedOpt_exprs = zip(*sorted(zip(losses, list(opt_exprs.keys()))))

        self.bestExpressions = []

        for i in range(0, len(sortedLosses)):
            if (sortedLosses[i] >= self.epsloss):
                break

            expr = sympy.simplify(sortedOpt_exprs[i])

            if (not expr in self.avoided_expr):
                self.bestExpressions.append((expr, sortedLosses[i]))

        i = 0

        while (len(self.bestExpressions) == 0 and i < len(sortedOpt_exprs)):
            if (not sortedOpt_exprs[i] in self.avoided_expr):
                self.bestExpressions = [(sympy.simplify(sortedOpt_exprs[i]), sortedLosses[i])]

            i += 1

        if (self.csv_filename):
            df = pd.DataFrame()

            df_sym_expr = []
            df_opt_expr = []
            df_loss = []
            df_sym_complexity = []
            df_opt_complexity = []

            for e in self.expressions:
                df_sym_expr.append(e.sym_expr)
                df_opt_expr.append(e.opt_expr)
                df_loss.append(e.loss)
                df_sym_complexity.append(expression_complexity(e.sym_expr, self.op_weights)["total_weight"])
                df_opt_complexity.append(expression_complexity(e.opt_expr, self.op_weights)["total_weight"])

            df["sym_expr"] = df_sym_expr
            df["opt_expr"] = df_opt_expr
            df["loss"] = df_loss
            df["sym_complexity"] = df_sym_complexity
            df["opt_complexity"] = df_opt_complexity

            df = df.sort_values(by = "loss")

            df.to_csv(self.csv_filename)

def convolve(x, y):
    return np.array([np.sum(np.convolve(x[:i], y[:i])) for i in range(1, len(x) + 1)])

def sym_conv(x, y):
    return sympy.sympify("conv(" + str(x) + ", " + str(y) + ")")

def sym_inv_(x):
    return sympy.sympify("inv_(" + str(x) + ")")

def num_inv_(x):
    return 1 / x

def sym_neg(x):
    return sympy.sympify("neg(" + str(x) + ")")

def sym_floor(x):
    return sympy.sympify("floor(" + str(x) + ")")

def sym_ceil(x):
    return sympy.sympify("ceil(" + str(x) + ")")

def sym_sqrt(x):
    return sympy.sympify("sqrt(" + str(x) + ")")

def sym_fmin(x, y):
    return sympy.sympify("fmin(" + str(x) + ", " + str(y) + ")")

def sym_fmax(x, y):
    return sympy.sympify("fmax(" + str(x) + ", " + str(y) + ")")
