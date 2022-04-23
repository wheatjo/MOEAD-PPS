
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.factory import get_problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize


problem = get_problem("ackley", n_var=10)


algorithm = DE(
    pop_size=100,
    sampling=LHS(),
    variant="DE/rand/1/bin",
    CR=0.3,
    dither="vector",
    jitter=False
)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))