import random
from functools import partial
from deap import base
from deap import tools
from deap import creator
import numpy as np
from deap import tools

random.seed(42)


def generate_cho(N_dec):
    return [N_dec]


gen_idx = partial(generate_cho, 10)

c = gen_idx()
a = tools.initIterate(list, gen_idx)  # doctest: +SKIP
print("c", c)