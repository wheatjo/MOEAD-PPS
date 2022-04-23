import numpy as np
import matplotlib.pyplot as plt
from pymoo.factory import get_problem

fig, ax = plt.subplots()


problem = get_problem("zdt1")

pop = np.random.random((100, 30))
res_obj = problem.evaluate(pop)


