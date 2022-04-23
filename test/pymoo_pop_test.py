
from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from pymoo.util.display import Display
from pymoo.visualization.scatter import Scatter

problem = get_problem("dtlz2")

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = MOEAD(
    ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7,
)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               save_history=True,
               verbose=True)

# get_visualization("scatter").add(res.F).show()
Scatter(legend=True).add(problem.pareto_front(), label="Pareto-front").add(res.F, label="Result").show()
