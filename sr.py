from multiprocessing import cpu_count, Manager, Pool
import numpy as np
import operator
import random
import sympy

def split_dict_equally(d, n):
    items = list(d.items())
    chunk_size = (len(items) + n - 1) // n

    return [dict(items[i*chunk_size:(i+1)*chunk_size]) for i in range(n)]

def eval_binary_combination(args):
    key1, key2, name, expressions, binary_operator, y, loss_func, maxloss, maxsymbols, verbose, eps, avoided_expr, foundBreak, subs_expr, shared_finished = args

    if (shared_finished.value):
        return None

    expr1, expr2 = key1, key2
    value1, value2 = expressions[expr1], expressions[expr2]
    x1, loss1 = value1
    x2, loss2 = value2

    try:
        new_expr = binary_operator(x1, x2)
        
        if (name.isalnum()):
            expr_str = f"{name}({expr1}, {expr2})"
        else:
            expr_str = f"({expr1}){name}({expr2})"

        sym_expr = sympy.sympify(expr_str)

        for key, value in subs_expr.items():
            sym_expr = sympy.simplify(sym_expr.subs(key, value))

        if (sym_expr in expressions):
            return None

        loss = loss_func(new_expr, y)

        if (loss < eps):
            if (not sym_expr in avoided_expr):
                if (verbose):
                    print("Found expression:", sym_expr)

                if (foundBreak):
                    shared_finished.value = True
        else:
            if (sym_expr in avoided_expr):
                return None

            loss += loss1 + loss2

        if (maxloss > 0 and loss > maxloss) or (maxsymbols > 0 and len(sym_expr.free_symbols) > maxsymbols):
            return None

        return (str(sym_expr), new_expr, loss)
    except Exception:
        return None

def mse_loss(x, y):
    return np.sum((x - y) ** 2)

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
                 eps = 1e-12,
                 avoided_expr = [],
                 subs_expr = {}):
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
        
        assert(self.eps > 0)

        self.expressions = []
        self.bestExpressions = []
        self.lastIteration = -1

    def predict(self, X, y, variable_names = []):
        self.lastIteration = -1
        self.expressions = []

        given_symbols = sympy.symbols(" ".join(variable_names))

        if (type(given_symbols) == sympy.Symbol):
            given_symbols = [given_symbols]

        default_symbols = sympy.symbols(" ".join("x" + str(i) for i in range(0, len(X))))

        if (type(default_symbols) == sympy.Symbol):
            default_symbols = [default_symbols]

        symbols = given_symbols

        if (len(symbols) < len(default_symbols)):
            symbols += default_symbols[len(given_symbols) - len(default_symbols):]

        expressions = {}

        for i in range(0, len(symbols)):
            expressions[symbols[i]] = (X[i], self.elementwise_loss(X[i], y))

        for j in range(0, self.niterations):
            if (self.verbose):
                print("Iteration #" + str(j))

            losses = [value[1] for key, value in expressions.items()]

            sortedLosses = sorted(losses)
            keys = list(expressions.keys())

            self.lastIteration = j

            finished = False

            newExpressions = {}

            for key, value in expressions.items():
                x, loss = value
                
                for name, unary_operator in self.unary_operators.items():
                    newExpr = sympy.sympify(name + "(" + str(key) + ")")
                    
                    if (not newExpr in expressions):
                        newx = unary_operator(x)

                        newLoss = self.elementwise_loss(newx, y)
                        
                        expr = sympy.simplify(newExpr)
                        
                        for key, value in self.subs_expr.items():
                            expr = sympy.simplify(expr.subs(key, value))

                        if (loss < self.eps and not expr in self.avoided_expr):
                            if (self.verbose):
                                print("Found expression:", expr)

                            if (self.foundBreak):
                                finished = True
                                break

                        newLoss += loss
                        
                        if ((self.maxloss > 0 and loss <= self.maxloss)
                            and not expr in self.avoided_expr):
                            newExpressions[str(expr)] = (newx, newLoss)

            expressions = {**expressions, **newExpressions}

            newExpressions = {}

            if (finished):
                break

            tasks = []
            groups = split_dict_equally(expressions, len(expressions) // self.group_expr_size + 1) if self.group_expr_size > 0 else [expressions]

            with Manager() as manager:
                shared_finished = manager.Value('b', False)
                
                for name, binary_operator in self.binary_operators.items():
                    for group in groups:
                        keys = list(group.keys())

                        indices1 = list(range(0, len(keys)))

                        if (self.shuffle_indices):
                            random.shuffle(indices1)

                        for i1 in indices1:
                            indices2 = list(range(i1 if name in self.symmetric_binary_operators else 0, len(keys)))

                            if (self.shuffle_indices):
                                random.shuffle(indices2)

                            for i2 in indices2:
                                tasks.append((keys[i1], keys[i2], name, expressions, binary_operator, y,
                                              self.elementwise_loss, self.maxloss, self.maxsymbols, self.verbose,
                                              self.eps, self.avoided_expr, self.foundBreak, self.subs_expr, shared_finished))

                with Pool(processes = cpu_count()) as pool:
                    results = pool.map(eval_binary_combination, tasks)

                finished = shared_finished.value

            for res in results:
                if (res is not None):
                    expr = sympy.simplify(res[0])
                    
                    newExpressions[str(expr)] = (res[1], res[2])

                    if (res[2] < self.eps and not expr in self.avoided_expr):
                        if (self.foundBreak):
                            finished = True
                            break

            if (self.discard_previous_expr):
                self.expressions += expressions.keys()
                expressions = {}

            expressions = {**expressions, **newExpressions}

            if (self.maxexpr > 0):
                while (len(expressions) > self.maxexpr):
                    del expressions[list(expressions.keys())[0]]
                    #del expressions[list(expressions.keys())[-1]]

            if (finished):
                break

        losses = [value[1] for key, value in expressions.items()]
        sortedLosses, sortedExpressions = zip(*sorted(zip(losses, [str(x) for x in expressions.keys()])))

        self.expressions += expressions

        self.bestExpressions = []

        for i in range(0, len(sortedLosses)):
            if (sortedLosses[i] >= self.eps):
                break

            expr = sympy.simplify(sortedExpressions[i])
            
            if (not expr in self.avoided_expr):
                self.bestExpressions.append(expr)

        if (len(self.bestExpressions) == 0):
            self.bestExpressions = [sympy.simplify(self.expressions[0])]

def convolve(x, y):
    return np.array([np.sum(np.convolve(x[:i], y[:i])) for i in range(1, len(x) + 1)])

def test1():
    model = SR(niterations = 5,
               unary_operators = {"-": operator.neg},
               binary_operators = {"+": operator.add, "-": operator.sub, "*": operator.mul},
               foundBreak = True,
               symmetric_binary_operators = ["+", "*", "conv"])
    #unary_operators = {"-": operator.neg, "abs": operator.abs,
    #                   "inv": lambda x: 1 / x,
    #                   "sqrt": lambda x: np.sqrt(x),
    #                   "cos": lambda x: np.cos(x),
    #                   "sin": lambda x: np.sin(x),
    #                   "ln": lambda x: np.log(x),
    #                   "exp": lambda x: np.exp(x),}
    #binary_operators = {"+": operator.add, "-": operator.sub,
    #                    "*": operator.mul, "/": operator.truediv, "//": operator.floordiv,
    #                    "%": operator.mod,
    #                    "conv": convolve,
    #                    "**": operator.pow}

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = x1 * x2

    model.predict(X, y, ["x1", "x2"])

    print("Model found in " + str(model.lastIteration + 1) + " iterations")
    print(model.bestExpressions)

def test2():
    model = SR(niterations = 5,
               unary_operators = {"-": operator.neg},
               binary_operators = {"+": operator.add, "-": operator.sub, "*": operator.mul},
               foundBreak = True,
               symmetric_binary_operators = ["+", "*", "conv"])

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = x1 + x2

    model.predict(X, y, ["x1", "x2"])

    print("Model found in " + str(model.lastIteration + 1) + " iterations")
    print(model.bestExpressions)

def test3():
    model = SR(niterations = 5,
               unary_operators = {"-": operator.neg},
               binary_operators = {"+": operator.add, "-": operator.sub, "*": operator.mul},
               foundBreak = True,
               symmetric_binary_operators = ["+", "*", "conv"])

    n = 10
    x1 = np.random.rand(n)
    x2 = np.random.rand(n)
    X = [x1, x2]
    y = (x1 - x2) ** 2 + x1 * x2

    model.predict(X, y, ["x1", "x2"])

    print("Model found in " + str(model.lastIteration + 1) + " iterations")
    print(model.bestExpressions)

def test4():
    model = SR(niterations = 5,
               binary_operators = {"-": operator.sub, "conv": convolve},
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

if (__name__ == "__main__"):
    import multiprocessing

    multiprocessing.freeze_support()

    test1()
    test2()
    test3()
    test4()
