from pymoo.factory import get_problem, get_reference_directions, get_visualization
from pymoo.util.plotting import plot
from pymoo.optimize import minimize
import numpy as np

import numpy as np
from pymoo.core.problem import ElementwiseProblem


class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


# problem = MyProblem()
problem = get_problem("dtlz1")
print(problem.n_var)
# x = np.random.random(2)
#
# x = x[np.newaxis, :]
# print(x)
x = np.vstack((np.random.random(2),np.random.random(2)))
print(x)
out = {"F":None, "G":None}
# can use pymoo benchmark to test algorithm
g = problem.evaluate(x,return_as_dictionary=True)
print(g)

