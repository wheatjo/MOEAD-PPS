from pymoo.factory import get_problem

from problem.LIRCMOP.lircmop1 import LIRCMOP1
import numpy as np
from pymoo.core.problem import calc_pf
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter

# test_problem = get_problem("c1dtlz1")
# ctp1 = get_problem("ctp1")
# pf = test_problem.pareto_front()
# pf_ctp1 = ctp1.pareto_front(use_cache=True)
# print(pf)
# print(pf_ctp1)


def create_pop(n=100, var=2):
    pop = np.random.random(((n, var)))
    return pop


lircmop1 = LIRCMOP1()
x = create_pop(100, 30)

res = lircmop1.evaluate(x, return_as_dictionary=True)
print(res)

pf = calc_pf(lircmop1)
print(pf)

plot = Scatter()
plot.add(pf, color="red")
plot.add(res["F"], color="blue")
plot.show()