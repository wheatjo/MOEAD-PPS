from deap import base
from deap import tools
from deap import creator
import numpy as np
import argparse
from pymoo.factory import get_problem


def parse_args():
    parser = argparse.ArgumentParser(description='args for MOEAD-PPS')
    parser.add_argument('--problem', type=str, default='dtlz1',
                        help='get problem from pymoo, or design problem in pymoo')
    parser.add_argument('--N', type=int, default=100, help='the number of population')
    args = parser.parse_args()
    return args


def generate_chromosome(N_dec=10):
    chromosome = np.random.random(N_dec)
    return chromosome


def evolve():
    args = parse_args()
    # how to let coder select problem(both pymoo and designed by self !!!
    problem = get_problem(args.problem)
    n_objs = problem.n_obj
    n_decs = problem.n_var
    weight_fitness_deap = ()
    for i in range(n_objs):
        weight_fitness_deap = weight_fitness_deap + (-1, )

    creator.create('MultiFitness', base.Fitness, weights=weight_fitness_deap)
    creator.create('Individual', np.ndarray, fitness=creator.MultiFitness, cv=float)
    toolbox = base.Toolbox()
    toolbox.register("make_number_dec", generate_chromosome, n_decs)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.make_number_dec)
    toolbox.register("popluation", tools.initRepeat, list, toolbox.individual)
    

if __name__ == '__main__':
    evolve()







