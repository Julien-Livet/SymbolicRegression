import copy
from deap import base, creator, tools, algorithms
import itertools
import math
import multiprocessing
from multiprocessing import cpu_count, Manager
import numpy as np
import operator
import random
from scipy.optimize import basinhopping, brute, curve_fit, differential_evolution
import sympy

def random_discrete_values(n, discrete_values):
    values = []

    for i in range(0, n):
        value = discrete_values[random.randint(0, len(discrete_values) - 1)]
        
        if (type(value) == str):
            s = value
            a, b = [float(x) for x in s[1:-1].split(",")]

            assert(a <= b)

            if (s[0] == "(" or s[0] == ")"):
                a, b = int(a), int(b)

                if (s[0] == ")"):
                    a += 1

                if (s[-1] == "("):
                    b -=1

                value = random.randint(a, b)
            elif (s[0] == "[" or s[0] == "]"):
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
                a, b = [float(x) for x in s[1:-1].split(",")]

                assert(a <= b)

                if (s[0] == "(" or s[0] == ")"):
                    a, b = int(a), int(b)

                    if (s[0] == ")"):
                        a += 1

                    if (s[-1] == "("):
                        b -=1

                if (a <= values[i] and values[i] <= b):
                    best_diff = 0
                    best_value = int(round(values[i]))

                    break
                else:
                    if (abs(values[i] - a) < best_diff):
                        best_diff = abs(values[i] - a)
                        best_value = a

                    if (abs(values[i] - b) < best_diff):
                        best_diff = abs(values[i] - b)
                        best_value = b
            else:
                if (abs(values[i] - v) < best_diff):
                    best_diff = abs(values[i] - v)
                    best_value = v

        values[i] = best_value

    return values

def fit(func, value_vars, y, p0, loss_func, eps, maxfev, discrete_values = []):
    if (len(discrete_values) == 0):
        try:
            value_params, _ = curve_fit(func, value_vars, y, p0 = p0, maxfev = maxfev)
        except RuntimeError as e:
            print(e)

            return p0

        return value_params

    try:
        value_params, _ = curve_fit(func, value_vars, y, p0 = p0, maxfev = maxfev)
    except RuntimeError as e:
        print(e)

        return p0

    best_x = random_discrete_values(len(p0), discrete_values)

    try:
        best_loss = loss_func(func(value_vars, *best_x), y)
        x = best_x

        for i in range(0, maxfev):
            try:
                value_params, _ = curve_fit(func, value_vars, y, p0 = random_discrete_values(len(p0), discrete_values), maxfev = 10 * maxfev)

                x = round_discrete_values(value_params, discrete_values)
                loss = loss_func(func(value_vars, *x), y)

                if (loss < best_loss and any(x)):
                    best_loss = loss
                    best_x = x

                    if (best_loss < eps):
                        break
            except RuntimeError:
                pass
            except ValueError:
                pass
            except OverflowError:
                pass
    except ValueError:
        pass
    except OverflowError:
        pass

    return best_x

def split_list(lst, n):
    k, m = divmod(len(lst), n)

    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def round_val(x, eps):
    r = round(x / eps) * eps

    if (abs(int(r) - r) <= eps):
        return int(r)
    else:
        return r

def sym_expr_eq(expr1, expr2, symbols = []):
    expr1 = sympy.expand(expr1)
    expr2 = sympy.expand(expr2)

    poly1 = sympy.Poly(expr1, *symbols)
    poly2 = sympy.Poly(expr2, *symbols)
    
    if (poly1 == None or poly2 == None):
        return False

    if (set(poly1.monoms()) != set(poly2.monoms())):
        return False

    return True

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

def mse_loss(x, y):
    return np.sum((x - y) ** 2)

def model_func(func):
    def model(x, *args):
        return np.array(func(*x, *args))

    return model

def new_params(expr, symbols):
    new_symbol_params = []
    new_value_params = []

    combined_symbols = set()

    poly = expr.as_poly(*symbols)
    degree = 0

    if (poly == None):
        return (new_symbol_params, new_value_params, expr, {})

    cst = poly.coeff_monomial(1)

    poly = poly - cst

    for s in symbols:
        degree = max(degree, poly.degree(s))

    for d in range(1, degree + 1):
        for comb in itertools.combinations_with_replacement(symbols, d):
            combined_symbols.add(sympy.Mul(*sorted(comb, key = str)))

    combined_symbols = list(combined_symbols)
    replacements = {}

    terms = []

    if (expr - cst == 0):
        terms = [0 for s in combined_symbols]
    else:
        p = sympy.Poly(expr - cst, combined_symbols)
        term_dict = p.as_dict()
        monomes = list(itertools.product(*[range(p.degree(v) + 1) for v in combined_symbols]))
        del monomes[0]
        
        monomes_symboliques = [
            sympy.Mul(*[var**exp for var, exp in zip(combined_symbols, exposants)])
            for exposants in monomes
        ]
        
        terms = [term_dict.get(exp, 0) for exp in monomes]
        
        for i in range(0, len(terms)):
            if (terms[i]):
                s = newSymbol()
                while (s in new_symbol_params):
                    s = newSymbol()
                new_symbol_params.append(s)
                new_value_params.append(1.0)
                replacements[terms[i]] = new_symbol_params[-1]
                terms[i] = new_symbol_params[-1]
        
        combined_symbols = monomes_symboliques

    if (cst != 0):
        combined_symbols.append(1)
        s = newSymbol()
        while (s in new_symbol_params):
            s = newSymbol()
        new_symbol_params.append(s)
        new_value_params.append(1.0)
        terms.append(new_symbol_params[-1])
        replacements[cst] = new_symbol_params[-1]

    expr = sum(c * m for c, m in zip(terms, combined_symbols))

    return (new_symbol_params, new_value_params, expr, replacements)

class Expr:
    def __init__(self, symbol_var = None, value_var = None, expr = None, symbol_vars = None, value_vars = None):
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

    def compute_opt_expr(self, y, loss_func, subs_expr, eps, unary_ops, binary_ops, maxfev, epsloss, fixed_cst_value = None, discrete_param_values = []):
        modules = ['numpy']

        for name, op in unary_ops.items():
            sym_op, num_op = op
            
            if (type(sym_op) == sympy.Function):
                modules.append({str(sym_op): num_op})
        
        for name, op in binary_ops.items():
            sym_op, num_op = op
            
            #if (type(sym_op) == sympy.core.function.UndefinedFunction):
            if (name.isalnum()):
                modules.append({str(sym_op): num_op})

        symbol_params = copy.deepcopy(self.symbol_params)
        value_params = list(copy.deepcopy(self.value_params))
        sym_expr = copy.deepcopy(self.sym_expr)

        if (fixed_cst_value != None):
            sym_expr = sym_expr.subs(symbol_params[-1], fixed_cst_value)
            del symbol_params[-1]
            del value_params[-1]

        try:
            f = sympy.lambdify(self.symbol_vars + symbol_params, sym_expr, modules = modules)
        except SyntaxError as e:
            print(e)

            return

        func = model_func(f)

        try:
            p0 = [float(x) for x in value_params]
            #p0 = np.random.randn(len(p0), 1)
            
            if (len(p0) <= len(y)):
                try:
                    value_params = fit(func, self.value_vars, y, p0, loss_func, eps, maxfev, discrete_param_values)
                except TypeError as e:
                    print(sym_expr)

                    print(e)

                    exit()

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

        y_pred = func(self.value_vars, *value_params)
        self.loss = loss_func(y_pred, y)
        self.opt_expr = sym_expr

        for i in range(0, len(symbol_params)):
            v = int(value_params[i]) if value_params[i] == int(value_params[i]) else value_params[i]
            self.opt_expr = self.opt_expr.subs(symbol_params[i], v)

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

        return expr

    def apply_binary_op(self, binary_sym_num_op, other_expr):
        s1 = copy.deepcopy(self.symbol_params)
        s2 = copy.deepcopy(other_expr.symbol_params)
        
        expr = copy.deepcopy(self)
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

        assert(len(expr.symbol_params) == len(expr.value_params))
        assert(len(expr.symbol_params) == len(set(expr.symbol_params)))
        
        expr.simplify()

        assert(len(expr.symbol_params) == len(expr.value_params))
        assert(len(expr.symbol_params) == len(set(expr.symbol_params)))
        
        return expr

    def simplify(self):
        sym_expr = sympy.expand(self.sym_expr)

        new_symbol_params = self.symbol_params
        new_value_params = self.value_params
        replacements = {}

        symbols = set(copy.deepcopy(self.symbol_vars))
        replaced_symbols = {}

        for node in sympy.preorder_traversal(sym_expr):
            if (isinstance(node, sympy.Function)):
                symbol = newSymbol()
                while (symbol in symbols):
                    symbol = newSymbol()
                symbols.add(symbol)
                replaced_symbols[symbol] = str(node.func) + "(" + ", ".join([str(x) for x in node.args]) + ")"
                replacements[node] = symbol

        symbols = list(symbols)

        n_p = new_params(sym_expr.xreplace(replacements), symbols)
        
        original_symbol_params = copy.deepcopy(new_symbol_params)
        original_value_params = copy.deepcopy(new_value_params)

        new_symbol_params = n_p[0]
        new_value_params = n_p[1]

        for k, v in replaced_symbols.items():
            try:
                i = new_symbol_params.index(k)
                del new_symbol_params[i]
                del new_value_params[i]
            except ValueError:
                pass

        self.sym_expr = n_p[2]
        replacements = n_p[3]

        for key, value in replaced_symbols.items():
            self.sym_expr = self.sym_expr.subs(key, value)

        self.symbol_params = copy.deepcopy(new_symbol_params)
        self.value_params = np.array(new_value_params)

        for k, v in replacements.items():
            try:
                i = original_symbol_params.index(k)
                del original_symbol_params[i]
                del original_value_params[i]
            except ValueError:
                pass

        free_symbols = self.sym_expr.free_symbols

        for i in range(0, len(original_symbol_params)):
            if (original_symbol_params[i] in free_symbols and not original_symbol_params[i] in self.symbol_params):
                self.symbol_params.append(original_symbol_params[i])
                self.value_params = np.array(list(self.value_params) + [original_value_params[i]])

        return self

def eval_binary_combination(args):
    expr1, expr2, name, opt_exps, binary_operator, y, loss_func, maxloss, maxsymbols, verbose, eps, epsloss, avoided_expr, foundBreak, subs_expr, un_ops, bin_ops, maxfev, fixed_cst_value, discrete_param_values, groupId, taskId, process_sym_expr, symbols, shared_finished = args

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

    new_expr = expr1.apply_binary_op(binary_operator, expr2)
    
    try:
        new_expr.compute_opt_expr(y, loss_func, subs_expr, eps, un_ops, bin_ops, maxfev, epsloss, fixed_cst_value, discrete_param_values)
        s = str(new_expr.opt_expr)
    except ZeroDivisionError:
        return None

    if (maxloss <= 0 or new_expr.loss <= maxloss):
        if (not new_expr.opt_expr in avoided_expr):
            if (new_expr.loss < epsloss and foundBreak):
                if (verbose):
                    print("Found expression:", str(new_expr.opt_expr))

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
                 maxloss = -1,
                 maxsymbols = -1,
                 maxexpr = -1,
                 discard_previous_expr = False,
                 symmetric_binary_operators = {"+": True, "-": True, "*": False, "conv": False, "%": False}, #True for strict symmetry
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
                 process_sym_expr = None):
        self.niterations = niterations
        self.unary_operators = unary_operators
        self.binary_operators = binary_operators
        self.elementwise_loss = elementwise_loss
        self.foundBreak = foundBreak
        self.maxloss = maxloss
        self.maxsymbols = maxsymbols
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
        
        assert(self.eps > 0)

        self.expressions = []
        self.bestExpressions = []
        self.lastIteration = -1

    def predict(self, X, y, variable_names = []):
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

        exprs = []
        opt_exprs = {}

        for i in range(0, len(symbols)):
            exprs.append(Expr(symbols[i], X[i]))
            exprs[-1].compute_opt_expr(y, self.elementwise_loss, self.subs_expr, self.eps, self.unary_operators,
                                       self.binary_operators, self.maxfev, self.maxloss, self.fixed_cst_value, self.discrete_param_values)
            opt_exprs[str(exprs[-1].opt_expr)] = exprs[-1].loss
            
        for ee in self.extra_start_sym_expr:
            exprs.append(Expr(expr = ee, symbol_vars = symbols, value_vars = X))
            exprs[-1].compute_opt_expr(y, self.elementwise_loss, self.subs_expr, self.eps, self.unary_operators,
                                       self.binary_operators, self.maxfev, self.maxloss, self.fixed_cst_value, self.discrete_param_values)
            opt_exprs[str(exprs[-1].opt_expr)] = exprs[-1].loss

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
            return

        self.expressions = []

        for j in range(0, self.niterations):
            if (self.verbose):
                print("Iteration #" + str(j))

            self.lastIteration = j

            finished = False

            newExprs = []

            for name, unary_operator in self.unary_operators.items():
                for expr in exprs:
                    process = True

                    if (self.process_sym_expr != None):
                        process = False
                        
                        for e in self.process_sym_expr:
                            if (sym_expr_eq(e, expr.sym_expr, symbols)):
                                process = True
                                break
                    
                    if (process):
                        new_expr = expr.apply_unary_op(unary_operator)
                        
                        try:
                            new_expr.compute_opt_expr(y, self.elementwise_loss, self.subs_expr, self.eps, self.unary_operators,
                                                      self.binary_operators, self.maxfev, self.maxloss, self.fixed_cst_value, self.discrete_param_values)

                            if (self.maxloss <= 0 or new_expr.loss <= self.maxloss):
                                if (not new_expr.opt_expr in self.avoided_expr):
                                    newExprs.append(new_expr)
                                    opt_exprs[str(new_expr.opt_expr)] = new_expr.loss
                                    
                                    if (new_expr.loss < self.epsloss):
                                        if (verbose):
                                            print("Found expression:", str(new_expr.opt_expr))

                                        if (self.foundBreak):
                                            finished = True
                                            break
                        except ZeroDivisionError:
                            pass
                
                if (finished):
                    break

            if (len(self.unary_operators)):
                if (self.discard_previous_expr):
                    exprs = newExprs
                else:
                    exprs += newExprs

                if (self.sort_by_loss):
                    exprs = sorted(exprs, key=lambda x: x.loss)

            if (self.verbose):
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

            with Manager() as manager:
                shared_finished = manager.Value('b', False)
                
                for name, binary_operator in self.binary_operators.items():
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
                                newTasks.append((group[i1], group[i2], name, opt_exprs, binary_operator, y,
                                                self.elementwise_loss, self.maxloss, self.maxsymbols, self.verbose,
                                                self.eps, self.epsloss, self.avoided_expr, self.foundBreak, self.subs_expr,
                                                self.unary_operators, self.binary_operators, self.maxfev, self.fixed_cst_value,
                                                self.discrete_param_values, groupId, len(tasks) - n + len(newTasks),
                                                self.process_sym_expr, symbols, shared_finished))

                        if (self.maxtask > 0):
                            newTasks = newTasks[:self.maxtask]

                        tasks += newTasks

                        if (self.verbose):
                            print("Operator " + name + " group #" + str(groupId) + " with " + str(len(tasks) - n) + " tasks")

                results = []

                shared_value = manager.Value('i', symbolIndex)

                with multiprocessing.Pool(initializer = init_shared, initargs = (shared_value,), processes = cpu_count()) as pool:
                #with multiprocessing.dummy.Pool(initializer = init_shared, initargs = (shared_value,), processes = cpu_count()) as pool:
                    results = pool.map(eval_binary_combination, tasks)
                #for t in tasks:
                #    results.append(eval_binary_combination(t))

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
                for k1 in range(0, len(self.checked_sym_expr)):
                    ce = self.checked_sym_expr[k1]
                    
                    for k2 in range(0, len(exprs)):
                        e = exprs[k2]

                        if (sym_expr_eq(e.sym_expr, ce, symbols)):
                            print("Checked expression", ce, k1, k2, e.opt_expr, e.loss)

            if (self.sort_by_loss):
                exprs = sorted(exprs, key = lambda x: x.loss)

            if (self.maxexpr > 0 and len(exprs) > self.maxexpr):
                exprs = exprs[:self.maxexpr]

            if (finished):
                break

        losses = [value for key, value in opt_exprs.items()]
        sortedLosses, sortedOpt_exprs = zip(*sorted(zip(losses, list(opt_exprs.keys()))))

        self.bestExpressions = []

        for i in range(0, len(sortedLosses)):
            if (sortedLosses[i] >= self.epsloss):
                break

            expr = sympy.simplify(sortedOpt_exprs[i])
            
            if (not expr in self.avoided_expr):
                self.bestExpressions.append((expr, sortedLosses[i]))

        if (len(self.bestExpressions) == 0):
            self.bestExpressions = [(sympy.simplify(sortedOpt_exprs[0]), sortedLosses[0])]

def convolve(x, y):
    return np.array([np.sum(np.convolve(x[:i], y[:i])) for i in range(1, len(x) + 1)])
