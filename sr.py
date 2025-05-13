import copy
import itertools
import math
from multiprocessing import cpu_count, Manager#, Pool
from multiprocessing.dummy import Pool
import numpy as np
import operator
import random
from scipy.optimize import curve_fit
import sympy

def split_list(lst, n):
    k, m = divmod(len(lst), n)

    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def expr_eq(expr1, expr2, subs_expr = {}, eps = 5e-5):
    expr = sympy.factor(sympy.sympify(expr1 - expr2))

    for key, value in subs_expr.items():
        expr = expr.subs(key, value)

    expr = sympy.simplify(expr)

    expr = expr.replace(
        lambda e: e.is_Number and abs(e.evalf()) < eps,
        lambda e: 0
    )

    return sympy.simplify(expr) == sympy.sympify("0")

def sign(num):
    return -1 if num < 0 else 1

symbolIndex = 0

def newSymbol():
    global symbolIndex

    s = sympy.Symbol("_" + str(symbolIndex))
    
    symbolIndex += 1
    
    return s

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
    cst = poly.coeff_monomial(1)

    poly = poly - cst

    for s in symbols:
        degree = max(degree, poly.degree(s))

    for d in range(1, degree + 1):
        for comb in itertools.combinations_with_replacement(symbols, d):
            combined_symbols.add(sympy.Mul(*sorted(comb, key=str)))

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
                new_symbol_params.append(newSymbol())
                new_value_params.append(1.0)
                replacements[terms[i]] = new_symbol_params[-1]
                terms[i] = new_symbol_params[-1]
        
        combined_symbols = monomes_symboliques

    combined_symbols.append(1)
    new_symbol_params.append(newSymbol())
    new_value_params.append(1.0)
    terms.append(new_symbol_params[-1])
    replacements[cst] = new_symbol_params[-1]

    expr = sum(c * m for c, m in zip(terms, combined_symbols))

    return (new_symbol_params, new_value_params, expr, replacements)

class Expr:
    def __init__(self, symbol_var, value_var):
        a = newSymbol()
        b = newSymbol()
        
        self.opt_expr = ""
        self.sym_expr = a * symbol_var + b
        self.symbol_vars = [symbol_var]
        self.value_vars = [value_var]
        self.symbol_params = [a, b]
        self.value_params = np.array([1.0, 0.0])
        self.loss = math.inf
    
    def compute_opt_expr(self, y, loss_func, subs_expr, eps, unary_ops, binary_ops, maxfev):
        modules = ['numpy']
        
        for name, op in unary_ops.items():
            sym_op, num_op = op
            
            if (type(sym_op) == sympy.Function):
                modules.append({str(sym_op): num_op})
        
        for name, op in binary_ops.items():
            sym_op, num_op = op
            
            if (type(sym_op) == sympy.core.function.UndefinedFunction):
                modules.append({str(sym_op): num_op})

        f = sympy.lambdify(self.symbol_vars + self.symbol_params, self.sym_expr, modules = modules)
        func = model_func(f)

        try:
            p0 = [float(x) for x in self.value_params]
            self.value_params, _ = curve_fit(func, self.value_vars, y, p0 = p0)

            for i in range(0, len(self.value_params)):
                self.value_params[i] = round(self.value_params[i] / eps) * eps

                if (abs(self.value_params[i]) < eps):
                    self.value_params[i] = 0
                    
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

        y_pred = func(self.value_vars, *self.value_params)
        self.loss = loss_func(y_pred, y)
        self.opt_expr = self.sym_expr

        for i in range(0, len(self.symbol_params)):
            v = int(self.value_params[i]) if self.value_params[i] == int(self.value_params[i]) else self.value_params[i]
            self.opt_expr = self.opt_expr.subs(self.symbol_params[i], v)

        self.opt_expr = sympy.factor(sympy.sympify(self.opt_expr))

        for key, value in subs_expr.items():
            self.opt_expr = sympy.simplify(self.opt_expr.subs(key, value))

        self.opt_expr = self.opt_expr.replace(
            lambda e: e.is_Number and abs(e.evalf()) < eps,
            lambda e: 0
        )

    def apply_unary_op(self, unary_sym_num_op):
        expr = copy.deepcopy(self)
        sym_op, num_op = unary_sym_num_op
        
        a = newSymbol()
        b = newSymbol()

        expr.sym_expr = a * sym_op(expr.sym_expr) + b
        expr.symbol_params = list(expr.symbol_params) + [a, b]
        expr.value_params = list(expr.value_params) + list(np.array([1.0, 0.0]))

        expr.simplify()

        return expr

    def apply_binary_op(self, binary_sym_num_op, other_expr):
        expr = copy.deepcopy(self)
        sym_op, num_op = binary_sym_num_op
        
        a = newSymbol()
        b = newSymbol()

        symbol_params = []
        other_sym_expr = other_expr.sym_expr

        for i in range(0, len(other_expr.symbol_params)):
            symbol_params.append(newSymbol())
            other_sym_expr = other_sym_expr.subs(other_expr.symbol_params[i], symbol_params[-1])

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
                symbols.add(symbol)
                replaced_symbols[symbol] = str(node.func) + "(" + ", ".join([str(x) for x in node.args]) + ")"
                replacements[node] = symbol

        symbols = list(symbols)

        n_p = new_params(sym_expr.xreplace(replacements), symbols)
        
        original_symbol_params = new_symbol_params
        original_value_params = new_value_params
        
        new_symbol_params = n_p[0]
        new_value_params = n_p[1]

        for k, v in replaced_symbols.items():
            try:
                del new_symbol_params[new_symbol_params.index(k)]
            except ValueError:
                pass

        self.sym_expr = n_p[2]
        replacements = n_p[3]

        for key, value in replaced_symbols.items():
            self.sym_expr = self.sym_expr.subs(key, value)

        self.symbol_params = new_symbol_params
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
            if (original_symbol_params[i] in free_symbols):
                self.symbol_params.append(original_symbol_params[i])
                self.value_params = np.array(list(self.value_params) + [original_value_params[i]])

        return self

def eval_binary_combination(args):
    expr1, expr2, name, opt_exps, binary_operator, y, loss_func, maxloss, maxsymbols, verbose, eps, avoided_expr, foundBreak, subs_expr, un_ops, bin_ops, maxfev, shared_finished = args

    if (shared_finished.value):
        return None

    new_expr = expr1.apply_binary_op(binary_operator, expr2)
    new_expr.compute_opt_expr(y, loss_func, subs_expr, eps, un_ops, bin_ops, maxfev)
    s = str(new_expr.opt_expr)

    if (maxloss <= 0 or new_expr.loss <= maxloss):
        if (not new_expr.opt_expr in avoided_expr):
            if (new_expr.loss < eps and foundBreak):
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
                 symmetric_binary_operators = [],
                 shuffle_indices = False,
                 verbose = False,
                 group_expr_size = -1,
                 eps = 5e-5,
                 avoided_expr = [],
                 subs_expr = {},
                 sort_by_loss = False,
                 maxfev = 100):
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
        self.avoided_expr = avoided_expr
        self.subs_expr = subs_expr
        self.sort_by_loss = sort_by_loss
        self.maxfev = maxfev
        
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
            exprs[-1].compute_opt_expr(y, self.elementwise_loss, self.subs_expr, self.eps, self.unary_operators, self.binary_operators, self.maxfev)
            opt_exprs[str(exprs[-1].opt_expr)] = exprs[-1].loss

        self.expressions = opt_exprs

        losses = [value for key, value in opt_exprs.items()]
        sortedLosses, sortedOpt_exprs = zip(*sorted(zip(losses, list(opt_exprs.keys()))))

        self.bestExpressions = []

        for i in range(0, len(sortedLosses)):
            if (sortedLosses[i] >= self.eps):
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
                    new_expr = expr.apply_unary_op(unary_operator)
                    new_expr.compute_opt_expr(y, self.elementwise_loss, self.subs_expr, self.eps, self.unary_operators, self.binary_operators, self.maxfev)

                    if (self.maxloss <= 0 or new_expr.loss <= self.maxloss):
                        if (not new_expr.opt_expr in self.avoided_expr):
                            newExprs.append(new_expr)
                            opt_exprs[str(new_expr.opt_expr)] = new_expr.loss
                            
                            if (new_expr.loss < self.eps):
                                if (verbose):
                                    print("Found expression:", str(new_expr.opt_expr))

                                if (self.foundBreak):
                                    finished = True
                                    break
                
                if (finished):
                    break
            
            if (len(self.unary_operators)):
                if (self.discard_previous_expr):
                    exprs = newExprs
                else:
                    exprs += newExprs

                if (self.sort_by_loss):
                    exprs = sorted(exprs, key=lambda x: x.loss)

            newExprs = []

            if (finished):
                break

            tasks = []
            groups = split_list(exprs, len(exprs) // self.group_expr_size + 1) if self.group_expr_size > 0 else [exprs]

            with Manager() as manager:
                shared_finished = manager.Value('b', False)
                
                for name, binary_operator in self.binary_operators.items():
                    for group in groups:
                        indices1 = list(range(0, len(group)))

                        if (self.shuffle_indices):
                            random.shuffle(indices1)

                        for i1 in indices1:
                            indices2 = list(range(i1 if name in self.symmetric_binary_operators else 0, len(group)))

                            if (self.shuffle_indices):
                                random.shuffle(indices2)

                            for i2 in indices2:
                                tasks.append((group[i1], group[i2], name, opt_exprs, binary_operator, y,
                                              self.elementwise_loss, self.maxloss, self.maxsymbols, self.verbose,
                                              self.eps, self.avoided_expr, self.foundBreak, self.subs_expr,
                                              self.unary_operators, self.binary_operators, self.maxfev, shared_finished))

                results = []

                #with Pool(processes = cpu_count()) as pool:
                with Pool() as pool:
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

            if (self.sort_by_loss):
                exprs = sorted(exprs, key=lambda x: x.loss)

            if (self.maxexpr > 0 and len(exprs) > self.maxexpr):
                exprs = exprs[:self.maxexpr]
                #exprs = exprs[len(exprs)-self.maxexpr:]

            if (finished):
                break

        losses = [value for key, value in opt_exprs.items()]
        sortedLosses, sortedOpt_exprs = zip(*sorted(zip(losses, list(opt_exprs.keys()))))

        self.bestExpressions = []

        for i in range(0, len(sortedLosses)):
            if (sortedLosses[i] >= self.eps):
                break

            expr = sympy.simplify(sortedOpt_exprs[i])
            
            if (not expr in self.avoided_expr):
                self.bestExpressions.append((expr, sortedLosses[i]))

        if (len(self.bestExpressions) == 0):
            self.bestExpressions = [sympy.simplify(sortedOpt_exprs[0], sortedLosses[0])]

def convolve(x, y):
    return np.array([np.sum(np.convolve(x[:i], y[:i])) for i in range(1, len(x) + 1)])

def test4():
    model = SR(niterations = 3,
               binary_operators = {"-": (operator.sub, operator.sub), 
                                   "conv": (sympy.Function("conv"), convolve)},
               foundBreak = True,
               symmetric_binary_operators = ["+", "*", "conv"])

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = convolve(x1, x2) - x1

    model.predict(X, y, ["x1", "x2"])

    print("Model found in " + str(model.lastIteration + 1) + " iterations")
    print(model.bestExpressions)

def test7():
    model = SR(niterations = 3,
               binary_operators = {"+": (operator.add, operator.add),
                                   "*": (operator.mul, operator.mul),
                                   "conv": (sympy.Function("conv"), convolve)},
               foundBreak = True,
               symmetric_binary_operators = ["+", "*", "conv"])

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = 0.1 * convolve(0.2 * x1 + x2, 0.3 * x1 - 0.4 * x2) + 0.5

    model.predict(X, y, ["x1", "x2"])

    print("Model found in " + str(model.lastIteration + 1) + " iterations")
    print(model.bestExpressions)
