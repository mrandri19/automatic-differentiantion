import unittest


class Expr(object):
    pass

    def eval(self, env):
        raise NotImplementedError

    def derive_symbolic(self, var):
        raise NotImplementedError

    def derive_forward(self, env, var):
        raise NotImplementedError

    def simplify(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


class NumberExpr(Expr):
    def __init__(self, num):
        super()
        self.num = num

    def __repr__(self):
        return f"{self.num}"

    def eval(self, env):
        return self.num

    def derive_symbolic(self, var):
        return NumberExpr(0)

    def simplify(self):
        return self

    def __eq__(self, other):
        if not isinstance(other, NumberExpr):
            return False
        return self.num == other.num

    def derive_forward(self, env, var):
        return (self.num, 0)


class AddExpr(Expr):
    def __init__(self, lhs, rhs):
        super()
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"({self.lhs} + {self.rhs})"

    def eval(self, env):
        return self.lhs.eval(env) + self.rhs.eval(env)

    def derive_symbolic(self, var):
        return AddExpr(self.lhs.derive_symbolic(var), self.rhs.derive_symbolic(var))

    def simplify(self):
        simplified_lhs = self.lhs.simplify()
        simplified_rhs = self.rhs.simplify()

        if simplified_lhs == NumberExpr(0):
            return simplified_rhs
        if simplified_rhs == NumberExpr(0):
            return simplified_lhs
        return AddExpr(simplified_lhs, simplified_rhs)

    def __eq__(self, other):
        if not isinstance(other, AddExpr):
            return False
        return (self.lhs == other.lhs) and (self.rhs == other.rhs)

    def derive_forward(self, env, var):
        lhs, lhs_prime = self.lhs.derive_forward(env, var)
        rhs, rhs_prime = self.rhs.derive_forward(env, var)
        return (lhs+rhs, lhs_prime+rhs_prime)


class TimesExpr(Expr):
    def __init__(self, lhs, rhs):
        super()
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"({self.lhs} * {self.rhs})"

    def eval(self, env):
        return self.lhs.eval(env) * self.rhs.eval(env)

    def derive_symbolic(self, var):
        return AddExpr(
            TimesExpr(self.lhs.derive_symbolic(var), self.rhs),
            TimesExpr(self.lhs, self.rhs.derive_symbolic(var))
        )

    def simplify(self):
        simplified_lhs = self.lhs.simplify()
        simplified_rhs = self.rhs.simplify()

        if simplified_lhs == NumberExpr(0):
            return NumberExpr(0)
        if simplified_rhs == NumberExpr(0):
            return NumberExpr(0)
        if simplified_lhs == NumberExpr(1):
            return simplified_rhs
        if simplified_rhs == NumberExpr(1):
            return simplified_lhs
        return TimesExpr(simplified_lhs, simplified_rhs)

    def __eq__(self, other):
        if not isinstance(other, TimesExpr):
            return False
        return (self.lhs == other.lhs) and (self.rhs == other.rhs)

    def derive_forward(self, env, var):
        lhs, lhs_prime = self.lhs.derive_forward(env, var)
        rhs, rhs_prime = self.rhs.derive_forward(env, var)
        return (lhs*rhs, lhs_prime*rhs+lhs*rhs_prime)


class VarExpr(Expr):
    def __init__(self, name):
        super()
        self.name = name

    def __repr__(self):
        return f"{self.name}"

    def eval(self, env):
        return env[self.name]

    def derive_symbolic(self, var):
        return NumberExpr(1 if var == self.name else 0)

    def simplify(self):
        return self

    def __eq__(self, other):
        if not isinstance(other, VarExpr):
            return False
        return self.name == other.name

    def derive_forward(self, env, var):
        return (env[self.name], 1 if self.name == var else 0)


class Tests(unittest.TestCase):
    def test_repr(self):
        self.assertEqual(repr(TimesExpr(NumberExpr(2), AddExpr(
            VarExpr("x"), NumberExpr(4)))), "(2 * (x + 4))")

    def test_eval(self):
        self.assertEqual(
            AddExpr(VarExpr("x"), NumberExpr(4)).eval({"x": 5}), 9)
        self.assertEqual(TimesExpr(NumberExpr(2), AddExpr(
            VarExpr("x"), NumberExpr(4))).eval({"x": 5}), 18)
        with self.assertRaises(KeyError):
            TimesExpr(NumberExpr(2), AddExpr(
                VarExpr("x"), NumberExpr(4))).eval({})

    def test_simplify(self):
        self.assertEqual(TimesExpr(NumberExpr(
            0), NumberExpr(4)).simplify(), NumberExpr(0))
        self.assertEqual(TimesExpr(NumberExpr(
            1), NumberExpr(4)).simplify(), NumberExpr(4))
        self.assertEqual(
            AddExpr(NumberExpr(0), NumberExpr(2)).simplify(), NumberExpr(2))

    def test_derive_symbolic(self):
        self.assertEqual(TimesExpr(NumberExpr(2), AddExpr(
            VarExpr("x"), NumberExpr(4))).derive_symbolic("x"), AddExpr(
                TimesExpr(NumberExpr(0), AddExpr(VarExpr("x"), NumberExpr(4))),
                TimesExpr(NumberExpr(2), AddExpr(
                    NumberExpr(1), NumberExpr(0)))))

        self.assertEqual(TimesExpr(NumberExpr(2), AddExpr(
            VarExpr("x"), NumberExpr(4))).derive_symbolic("x").simplify(), NumberExpr(2))

    def test_derive_computational(self):
        expr = TimesExpr(NumberExpr(2), AddExpr(VarExpr("x"), NumberExpr(4)))

        for i in range(5):
            h = 10 ** -i
            self.assertAlmostEqual(
                (expr.eval({"x": 1+h})-expr.eval({"x": 1}))/h, 2)

    def test_derive_forward(self):
        expr = TimesExpr(NumberExpr(2), AddExpr(VarExpr("x"), NumberExpr(4)))
        self.assertEqual(expr.eval({"x": 1}), 10)
        self.assertEqual(expr.derive_forward({"x": 1}, "x"), (10, 2))
        self.assertEqual(expr.derive_forward({"x": 1}, "y"), (10, 0))
        self.assertEqual(expr.derive_forward({"x": 2}, "x"), (12, 2))


if __name__ == '__main__':
    unittest.main()
