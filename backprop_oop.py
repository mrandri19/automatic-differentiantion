from pprinter import PPrinter
import math

from graphviz import Graph


def make_graph():
    G = Graph()
    fontname = "Roboto Mono"
    G.attr("graph", rankdir="RL")
    G.attr("graph", fontname=fontname)
    G.attr("node", fontname=fontname)
    G.attr("node", style="rounded")
    G.attr("node", shape="box")
    G.attr("edge", fontname=fontname)
    return G


class IDManager:
    def __init__(self):
        self._id_counter = 1
        self.id_to_node = {}

    def new_id(self, node):
        new_id = f"w{self._id_counter}"
        self.id_to_node[new_id] = node
        self._id_counter += 1
        return new_id


pp = PPrinter()
idm = IDManager()


class Op:
    def __init__(self):
        self.id = idm.new_id(self)
        self.parents = []

        self.fwd_val = None

    def __repr__(self):
        return self.id

    def __mul__(self, other):
        return MulOp(self, other)

    def __add__(self, other):
        return AddOp(self, other)


class NullaryOp(Op):
    def __init__(self):
        super().__init__()

        self.adjoint = 0

    def graph(self, G):
        G.node(
            self.id, self.node_repr())

    def node_repr(self):
        return (f"<<font color=\"blue\">{self.id}</font><br/>"
                f"{self.__class__.__name__}<br/>"
                f"fwd: {self.fwd_val:.2f}<br/>>")

    def backward_local_grad(self):
        pass

    def backward(self, parent_adjoint):
        self.adjoint += parent_adjoint


class UnaryOp(Op):
    def __init__(self, arg):
        super().__init__()

        arg.parents.append(self)

        self.arg = arg

        self.darg = None

    def graph(self, G):
        G.node(
            self.id, self.node_repr())
        G.edge(self.id, self.arg.id, self.edge_repr())
        self.arg.graph(G)

    def node_repr(self):
        return (f"<<font color=\"blue\">{self.id}</font><br/>"
                f"{self.__class__.__name__}<br/>"
                f"fwd: {self.fwd_val:.2f}<br/>>")

    def edge_repr(self):
        return(f"\u2202{self.id}/\u2202{self.arg.id}: {self.darg:.2f}")


class BinaryOp(Op):
    def __init__(self, lhs, rhs):
        super().__init__()

        lhs.parents.append(self)
        rhs.parents.append(self)

        self.lhs = lhs
        self.rhs = rhs

        self.dlhs = None
        self.drhs = None

    def graph(self, G):
        G.node(
            self.id, self.node_repr())
        G.edge(self.id, self.lhs.id, self.left_edge_repr())
        G.edge(self.id, self.rhs.id, self.right_edge_repr())
        self.rhs.graph(G)
        self.lhs.graph(G)

    def node_repr(self):
        return (f"<<font color=\"blue\">{self.id}</font><br/>"
                f"{self.__class__.__name__}<br/>"
                f"fwd: {self.fwd_val:.2f}<br/>>")

    def right_edge_repr(self):
        return f"<\u2202{self.id}/\u2202{self.rhs.id}: {self.drhs:.2f}>"

    def left_edge_repr(self):
        return f"<\u2202{self.id}/\u2202{self.lhs.id}: {self.dlhs:.2f}>"

    def backward(self, parent_adjoint):
        self.lhs.backward(self.dlhs * parent_adjoint)
        self.rhs.backward(self.drhs * parent_adjoint)


class Number(NullaryOp):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def node_repr(self):
        return (f"<<font color=\"blue\">{self.id}</font><br/>"
                f"{self.__class__.__name__}: {self.value}<br/>"
                f"fwd: {self.fwd_val:.2f}<br/>>")

    def forward(self, env):
        self.fwd_val = self.value
        return self.fwd_val


class Var(NullaryOp):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def node_repr(self):
        return (f"<<font color=\"blue\">{self.id}</font><br/>"
                f"{self.__class__.__name__}: {self.name}<br/>"
                f"fwd: {self.fwd_val:.2f}<br/>>")

    def forward(self, env):
        self.fwd_val = env[self.name]
        return self.fwd_val


class ExpOp(UnaryOp):
    def __init__(self, arg):
        super().__init__(arg)

    def forward(self, env):
        self.fwd_val = math.exp(self.arg.forward(env))
        return self.fwd_val

    def backward_local_grad(self):
        self.darg = math.exp(self.arg.fwd_val)
        self.arg.backward_local_grad()

    def backward(self, parent_adjoint):
        self.arg.backward(self.darg * parent_adjoint)


def exp(arg):
    return ExpOp(arg)


class MulOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)

    def forward(self, env):
        self.fwd_val = self.lhs.forward(env) * self.rhs.forward(env)
        return self.fwd_val

    def backward_local_grad(self):
        self.dlhs = self.rhs.fwd_val
        self.drhs = self.lhs.fwd_val
        self.lhs.backward_local_grad()
        self.rhs.backward_local_grad()


class AddOp(BinaryOp):
    def __init__(self, lhs, rhs):
        super().__init__(lhs, rhs)

    def forward(self, env):
        self.fwd_val = self.lhs.forward(env) + self.rhs.forward(env)
        return self.fwd_val

    def backward_local_grad(self):
        self.dlhs = 1
        self.drhs = 1
        self.lhs.backward_local_grad()
        self.rhs.backward_local_grad()


# Define the expression
x1 = Var("x1")
x2 = Var("x2")
z = exp(x1*x2 + Number(5)*x1)

# Run the computation, saving the computed values at each node.
assert (abs(z.forward({"x1": 0.2, "x2": 0.3}) - math.exp(1.06)) <= 5*1e-15)

# Traverse the computation DAG backwards, calculating the adjoint at each edge.
# NOTE(Andrea): this could be done in the forward pass but, in general, not in
# the initialization (e.g. to make the adjoint for the multiplication we need
# to know the terms which are being multiplied).
z.backward_local_grad()

# Traverse the computation DAG backwards, at each edge multiplying the parent's
# adjoint with the edge's adjoint.
z.backward(1)

G = make_graph()
z.graph(G)
G.render(view=True, format="svg")
