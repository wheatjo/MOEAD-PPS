from pymoo.factory import get_problem, get_reference_directions, get_visualization
from pymoo.util.plotting import plot
from pymoo.optimize import minimize
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance

ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=91)
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


problem = MyProblem()
# problem = get_problem("zdt1")
# print(problem.n_obj)
# x = np.random.random(2)
#
# x = x[np.newaxis, :]
# print(x)
nvar = 2
x = np.vstack((np.random.random(nvar),np.random.random(nvar)))
x = np.vstack((x, np.random.random(nvar)))

def create_pop(n=100, var=2):
    pop = np.random.random(((n, var)))
    return pop

x = create_pop(100 ,2)
# out = {"F":None, "G":None}
# # can use pymoo benchmark to test algorithm
g = problem.evaluate(x,return_as_dictionary=True)
g_non = problem.evaluate(x)
print(g)
g_obj = g['F']
print(g_obj)
non_sort = NonDominatedSorting()

dis = calc_crowding_distance(g_obj)
# g_obj[1] = np.zeros(3)
frontsa = non_sort.do(g_obj)

print(frontsa)
print(len(dis))

dscend_order = np.argsort(dis)[::-1]
tt = np.sort(dis)

print(tt)

print(ref_dirs.shape)


