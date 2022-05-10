from pymoo.algorithms.moo.moead import MOEAD, ParallelMOEAD
from pymoo.factory import get_problem, get_visualization, get_reference_directions
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.display import Display
problem = get_problem("dtlz2")

ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

algorithm = ParallelMOEAD(
    ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7,
)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

get_visualization("scatter").add(res.F).show()