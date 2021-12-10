from deap import base
from deap import tools
from deap import creator
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='args for MOEAD-PPS')
    parser.add_argument('--problem', type=str, default='dtlz1',
                        help='get problem from pymoo, or design problem in pymoo')
    parser.add_argument('--N', type=int, default=100, help='the number of population')

    parser.add_argument('--have_design_problem', type=bool, default=False)
    args = parser.parse_args()
    return args



def generate_cho(N_dec=10):
    cho = np.random.random(N_dec)
    return cho



BOUND_LOW = 0.0
BOUND_UP = 1.0

creator.create('MultiFitness', base.Fitness, weights=(-1, -1, -1))
creator.create("Individual", np.ndarray, fitness=creator.MultiFitness, cv=float)

toolbox = base.Toolbox()
# register function ! you can define by yourself
toolbox.register("make_number_dec", generate_cho, 7)
# toolbox.register("individual", tools.initIterate, creator.Individual, partial(np.random.random, 10))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.make_number_dec)
toolbox.register("popluation", tools.initRepeat, list, toolbox.individual)
# inpdb is the probability of every bit in chromosome mutate, set 1/N_dec generally
# toolbox.register("mutation", tools.mutPolynomialBounded, eta=20.0, low=BOUND_LOW, up=BOUND_UP, indpb=1.0 / 30.0)
# toolbox.register("crossover", tools.cxUniform)
# toolbox.register("select", tools.selRandom)

from class_use_tools import *

pop_test = TestClassUseDeap()

aa = pop_test.DE_evolve(pop_test.pop)
print(aa)