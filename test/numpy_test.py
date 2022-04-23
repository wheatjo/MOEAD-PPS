
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_problem, get_reference_directions

F = get_problem("zdt3").pareto_front()
fc = get_problem("ctp1").pareto_front(use_cache=False, flatten=False)

plot = Scatter()

plot.add(F, s=5, facecolors='red', edgecolors='red')
plot.add(fc, s=30, facecolors='red', edgecolors='blue')
plot.show()