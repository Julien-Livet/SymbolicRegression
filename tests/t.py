import sympy

def sym_expr_eq(expr1, expr2, symbols = []):
    try:
        expr1 = sympy.expand(expr1)
        expr2 = sympy.expand(expr2)

        poly1 = sympy.Poly(expr1, *symbols)
        poly2 = sympy.Poly(expr2, *symbols)

        if (poly1 == None or poly2 == None):
            return False

        if (set(poly1.monoms()) != set(poly2.monoms())):
            return False

        for i in range(0, len(poly1.coeffs())):
            if (poly1.coeffs()[i].is_Number and poly2.coeffs()[i].is_Number):
                if (poly1.coeffs()[i] != poly2.coeffs()[i]):
                    return False

        return True
    except sympy.PolynomialError:
        diff = sympy.simplify(expr1 - expr2)
        
        sol = sympy.solve(diff, symbols, dict = True)
        
        print(sol)

def test_sym_expr_eq_1():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = 1
    expr2 = 2

    assert(not sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_2():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = 1
    expr2 = d

    assert(sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_3():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')
    
    expr1 = b + 1
    expr2 = d - 2

    assert(sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_4():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = a*x + b
    expr2 = c*x + d

    assert(sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_5():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = a*x + b
    expr2 = c*y + d

    assert(not sym_expr_eq(expr1, expr2, [x, y]))

def test_sym_expr_eq_6():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = (1+a)*x + b + 3
    expr2 = (2+c)*x + d

    assert(sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_7():
    a, b, c, d, x, y = sympy.symbols('a b c d x y')

    expr1 = x + a*x + b + 3
    expr2 = 2*x + c*x + d - 1

    assert(sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_8():
    a, b, c, d, e, f, x, y = sympy.symbols('a b c d e f x y')

    expr1 = a*x**2 + 3*x + c
    expr2 = d*x**2 + 2*x + f

    assert(not sym_expr_eq(expr1, expr2, [x]))

def test_sym_expr_eq_9():
    a, b, c, d, e, f, x, y = sympy.symbols('a b c d e f x y')

    expr1 = sympy.sympify("a*log(b*x+c)+d")
    expr2 = sympy.sympify("e+log(g+x*h)*f")

    assert(sym_expr_eq(expr1, expr2, [x]))
