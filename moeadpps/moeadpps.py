from deap import base
from deap import tools
from deap import creator
import numpy as np
import math
from utils.uniform_vector import uniform_vector_nbi
from sklearn.metrics.pairwise import euclidean_distances


class Individual(np.ndarray):

    def __new__(cls, chromosome, n_objs, n_cv):
        ind = np.asarray(chromosome).view(cls)
        ind.fitness = np.zeros(n_objs)
        ind.cv = np.zeros(n_cv)
        return ind


class MoeadPPS(object):

    def __init__(self, N, delta, nr, problem, max_function_evaluate):
        self.N = N
        self.delta = delta
        self.nr = nr
        self.problem = problem
        self.n_objs = problem.n_obj
        self.n_var = problem.n_var
        self.n_cv = problem.n_constr
        self.mutation_eta = 20.0
        self.var_up_bound = 1.0
        self.var_low_bound = 0.0
        self.max_gen = max_function_evaluate * 10
        self.max_function_evaluate = max_function_evaluate
        self.Tc = 0.9 * math.ceil(max_function_evaluate/self.N)
        self.last_gen = 20
        self.change_threshold = 0.1
        self.search_stage = 1
        self.max_change = 1
        self.epsilon_k = 0
        self.epsilon_0 = 0
        self.alpha = 0.95
        self.tao = 0.05
        self.T = math.ceil(N/problem.n_var) # the number of neighbour vector
        self.w_lambda_vector = uniform_vector_nbi(self.N, problem.n_obj)
        self.neighbour_w_ind = self.get_neighbour_vector(self.w_lambda_vector)
        self.ideal_point = np.zeros((1, problem.n_obj))
        self.nadir_point = np.zeros((1, problem.n_obj))
        self.pop = self.create_pop(self, N, problem.n_var, problem.n_obj, self.n_cv)
        self.crossover = tools.cxUniform
        self.mutation = tools.mutPolynomialBounded
        self.select = tools.selRandom

    def get_neighbour_vector(self, w_lambda):
        w_euclidean_distance = euclidean_distances(w_lambda, w_lambda)
        sort_ind = np.argsort(w_euclidean_distance)
        neighbours_vec = sort_ind[:, 0: self.T]
        return neighbours_vec

    @staticmethod
    def create_pop(self, N, n_dec, n_objs, n_cv):
        pop = []
        for i in range(N):
            chromosome = np.random.random(n_dec)
            pop.append(Individual(chromosome, n_objs, n_cv))

        return pop

    def differential_evolution_operator(self, parents: list):
        offspring_chromosome = parents[0] + 0.5 * (parents[1] - parents[2])
        offspring_chromosome = self.mutation(offspring_chromosome, eta=self.mutation_eta,
                                             low=self.var_low_bound, up=self.var_up_bound, indpb=1.0 / self.n_var)
        offspring = Individual(offspring_chromosome, self.n_var)

        return offspring

    def get_pop_cv_matrix(self):
        cv_matrix = np.zeros((self.N, self.n_cv)
        for i, ind in enumerate(self.pop):
            cv_matrix[i] = ind.cv

        return cv_matrix

    def evolution(self):
        gen = 1
        pop_cv = self.get_pop_cv_vector()



if __name__ == '__main__':
    moeadpps_objetc = MoeadPPS(100, 0.9, 2, )



