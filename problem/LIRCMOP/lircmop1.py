from pymoo.core.problem import Problem
import numpy as np


class LIRCMOP(Problem):

    def __init__(self, n_vars=None):

        if n_vars is not None:
            self.n_var = n_vars
            super().__init__(n_var=n_vars, n_obj=2, n_constr=2, x1=0.0, xu=1.0)
        else:
            self.n_var = 30
            super().__init__(n_var=30, n_obj=2, n_constr=2, x1=0.0, xu=1.0)

        self.J1 = np.arange(self.n_var)[2::2]
        self.J2 = np.arange(self.n_var)[1::2]
        self.J = np.arange(self.n_var)[2:-1]

    def calc_obj_value(self, x):
        pass

    def calc_constraint(self, x):
        pass


class LIRCMOP1(LIRCMOP):

    def __init__(self, n_vars=30):
        super(LIRCMOP1, self).__init__(n_vars)

    def _evaluate(self, x, out, *args, **kwargs):
        x_odd = x[:, self.J1]
        x_even = x[:, self.J2]
        g1 = np.sum(np.square(np.subtract(x_odd, np.sin(0.5*np.pi*x[:, 0]).reshape(x.shape[0], 1))), axis=1)
        g2 = np.sum(np.square(np.subtract(x_even, np.cos(0.5*np.pi*x[:, 0].reshape(x.shape[0], 1)))), axis=1)
        f1 = x[:, 0] + g1
        f2 = 1 - np.square(x[:, 0]) + g2
        c1 = (0.5 - g1) * (0.51 - g1)
        c2 = (0.51 - g2) * (0.5 - g2)
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([c1, c2])

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        f1 = x
        f2 = 1 - np.square(x)
        front = np.column_stack([f1, f2])+0.5
        return front

