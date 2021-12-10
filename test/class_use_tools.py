from deap import base
from deap import tools
from deap import creator
import numpy as np
from functools import partial
from pymoo.factory import get_problem
from pymoo.factory import get_mutation
from pymoo.interface import mutation
import random

# def generate_chromosome(N_dec=10):
#     chromosome = np.random.random(N_dec)
#     return chromosome
#
#
#
# BOUND_LOW = 0.0
# BOUND_UP = 1.0
#
# creator.create('MultiFitness', base.Fitness, weights=(-1, -1, -1))
# creator.create("Individual", np.ndarray, fitness=creator.MultiFitness, cv=float)
#
#toolbox = base.Toolbox()
# # register function ! you can define by yourself
# toolbox.register("make_number_dec", generate_chromosome, 7)
# # toolbox.register("individual", tools.initIterate, creator.Individual, partial(np.random.random, 10))
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.make_number_dec)
# toolbox.register("popluation", tools.initRepeat, list, toolbox.individual)
# # inpdb is the probability of every bit in chromosome mutate, set 1/N_dec generally
# toolbox.register("mutation", tools.mutPolynomialBounded, eta=20.0, low=BOUND_LOW, up=BOUND_UP, indpb=1.0 / 30.0)
# toolbox.register("crossover", tools.cxUniform)
# toolbox.register("select", tools.selRandom)


class Individual(object):

    def __init__(self, chromosome: np.ndarray, fitness: np.ndarray, cv: float):
        self.chromosome = chromosome
        self.fitness = fitness
        self.cv = cv

    def __add__(self, ind2):
        return Individual(self.chromosome + ind2.chromosome, np.zeros(self.fitness.shape[0]), 0)

    def __sub__(self, ind2):
        return Individual(self.chromosome - ind2.chromosome, np.zeros(self.fitness.shape[0]), 0)


class TestClassUseDeap(object):

    def __init__(self):

        self.ind = toolbox.individual()
        self.pop = toolbox.popluation(n=10)
        self.select_func = tools.selRandom



    def DE_evolve(self, pop):
        parent = self.select_func(pop, 3)
        offspring = toolbox.individual()
        offspring_cho = parent[0] + 0.5 * (parent[1] - parent[2])
        for i in range(len(offspring_cho)):
            offspring[i] = offspring_cho[i]

        mutant = toolbox.clone(offspring)
        offspring2, = toolbox.mutation(individual=mutant)
        return offspring


def add_1(x):
    return x+1


class Fitness(np.ndarray):
    def __new__(cls, fitness_value):
        fitness = np.asarray(fitness_value).view(cls)
        return fitness

    def __hash__(self):
        return random.randint(1,3)


class Individualnp(np.ndarray):

    def __new__(cls,n_decs, n_objs , cv=0):
        ind = np.asarray(np.random.random(n_decs)).view(cls)
        # ind = np.random.random(n_decs)
        ind.fitness = Fitness(np.zeros(n_objs))
        ind.cv = cv
        return ind


class IndividualInit(np.ndarray):

    def __new__(cls, chromosome, n_objs, cv=0):
        ind = np.asarray(chromosome).view(cls)
        ind.fitness = np.zeros(n_objs)
        ind.cv = cv
        return ind


ind_test = Individualnp(10,3, 1)
print(ind_test)
print(ind_test.fitness)
print(ind_test.cv)
ind_test2 = Individualnp(10,3,2)
print(ind_test2)
print(ind_test2.fitness)
print(ind_test2.cv)
offspring_chromosome = ind_test2 - ind_test
a = Fitness(np.array([1,2]))
offspring = IndividualInit(offspring_chromosome, 3)
print(offspring)
print(offspring.fitness)

test_pop = [ind_test, ind_test2, offspring]
print(test_pop)
test_pop_mat = np.array(test_pop)
print(test_pop_mat)
for i in test_pop:
    a = i.fitness
    print(a)
print("end")
choose = tools.selNSGA2(test_pop, len(test_pop))
print(choose)

def a(i):
    print(i.fitness)


a(offspring)