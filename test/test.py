from deap import base
from deap import tools
from deap import creator
import numpy as np
from functools import partial
from pymoo.factory import get_problem
from pymoo.factory import get_mutation
from pymoo.interface import mutation
# from utils.uniform_vector import Mean_vector

BOUND_LOW = 0.0
BOUND_UP = 1.0


def generate_cho(N_dec=10):
    cho = np.random.random(N_dec)
    return cho

a = (-1,-1,-1)
# creator.create('MultiFitness', base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create('MultiFitness', base.Fitness, weights=a)
creator.create("Individual", np.ndarray, fitness=creator.MultiFitness, cv=float)

toolbox = base.Toolbox()
# register function ! you can define by yourself
toolbox.register("make_number_dec", generate_cho, 7)
# toolbox.register("individual", tools.initIterate, creator.Individual, partial(np.random.random, 10))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.make_number_dec)
toolbox.register("popluation", tools.initRepeat, list, toolbox.individual)
# inpdb is the probability of every bit in chromosome mutate, set 1/N_dec generally
toolbox.register("mutation", tools.mutPolynomialBounded, eta=20.0, low=BOUND_LOW, up=BOUND_UP, indpb=1.0/30.0)
toolbox.register("crossover", tools.cxUniform)
ind_test = toolbox.individual()
print("fin: ", ind_test.fitness.valid)
pop_test = toolbox.popluation(n=15)
# ind_test.cv = 9.0
print(ind_test.cv)
print(ind_test.fitness.valid)
print(ind_test)

pop_matrix = np.array(pop_test)
print(pop_matrix.shape)

toolbox.register("select", tools.selRandom)

# select_parent = toolbox.select(pop_test, 3)
# print(select_parent)


def de_operator(parent):
    print("parent fitness", parent[0].fitness.value)
    offspring = toolbox.individual()
    offspring_cho = parent[0] + 0.5 * (parent[1]-parent[2])
    for i in range(len(offspring_cho)):
        offspring[i] = offspring_cho[i]

    mutant = toolbox.clone(offspring)
    offspring2,  = toolbox.mutation(individual=mutant)

    # print("is", offspring2 is offspring)
    # for i, data in enumerate(offspring):
    #     if(data != offspring2[i]):
    #         print("d", i)
    # del offspring2.fitness.values
    return offspring


def de_pymoo_operator(parent):
    offspring_ = parent[0] + 0.5 * (parent[1] - parent[2])
    # offspring_2 = get_mutation()
    off = mutation(get_mutation("real_pm", eta=20, prob=1.0), offspring_, 1)



# offspring = de_operator(select_parent)


problem = get_problem("dtlz1")
g = problem.evaluate(pop_matrix)
print("pip fiteness")
for i in range(g.shape[0]):
    pop_test[i].fitness.value = g[i,:]
    print(pop_test[i].fitness.value)

# print(pop_test[0].fitness.value)

select_parent = toolbox.select(pop_test, 3)
print(select_parent)
print(select_parent[0][1])
offspring = de_operator(select_parent)

print("offspring", offspring)

print(pop_test[0].fitness.value)
print(type(pop_test[0].fitness.value) == np.ndarray)


def make_fitness_matrix(pop):
    fit_mat = np.ndarray((len(pop), pop[0].fitness.value.shape[0]))
    for index, ind in enumerate(pop):
        fit_mat[index, :] = pop[index].fitness.value

    return fit_mat


fitness_mat = make_fitness_matrix(pop_test)

print(fitness_mat)

generate_vec = Mean_vector(H=3, m=3)
print(generate_vec.get_mean_vectors())


P = np.arange(len(pop_test)-8)
np.random.shuffle(P)
print("P", P)
W = np.array(generate_vec.get_mean_vectors())
print("W", W[P])
W_to_ind = fitness_mat[P]
print(W_to_ind)
g_mat = W[P]*W_to_ind[P]
print(g_mat)

g_value = np.max(g_mat, 1)
print(g_value)

fit_vector = problem.evaluate(offspring)
offspring.fitness.value = fit_vector

g_off = np.max(np.repeat(np.array([fit_vector]),len(P), axis=0)*W[P], 1)
print(g_off)









# ind_test.fitness.values = g
# print(np.max(ind_test.fitness.values))
# print("cv: ", ind_test.cv)

# def get_ideal_point(pop):





