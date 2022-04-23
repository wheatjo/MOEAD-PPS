import sys
import os
from deap import base
from deap import tools
from deap import creator
import numpy as np
import math

from sklearn.metrics.pairwise import euclidean_distances
from pymoo.core.problem import ElementwiseProblem
import copy
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance
from pymoo.factory import get_reference_directions, get_problem, get_visualization
from pymoo.interface import mutation
from pymoo.factory import get_mutation
from pymoo.visualization.scatter import Scatter
import logging
# from pymoo.core.problem import calc_pf

# from utils.savedata import save_data
# from pymoo.algorithms.soo.nonconvex.de import DE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

saveData = True

class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


class Individual(np.ndarray):

    def __new__(cls, chromosome, n_objs, n_cv):
        ind = np.asarray(chromosome).view(cls)
        ind.fitness = np.zeros(n_objs)
        ind.cv = np.zeros(n_cv)
        # ind.cv = np.zeros(n_cv)
        # ind.feasible = False
        return ind

    def copy(self, deep=False):
        ind_copy = Individual(self, self.fitness.shape[0], self.cv.shape[0])
        ind_copy.fitness = self.fitness
        ind_copy.cv = self.cv
        return ind_copy

    def __deepcopy__(self, memo):
        return self.copy(deep=True)


class MoeadPPS(object):

    def __init__(self, N, problem, max_function_evaluate, delta = 0.9, nr = 2):
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
        self.max_gen = math.ceil(max_function_evaluate/self.N)# max_function_evaluate * 10
        print("max gen: ", self.max_gen)
        self.max_function_evaluate = max_function_evaluate
        self.Tc = math.ceil(0.9 * math.ceil(max_function_evaluate/self.N))
        self.last_gen = 20
        self.change_threshold = 1e-3
        self.search_stage = 1
        self.max_change = 1
        self.epsilon_k = 0
        self.epsilon_0 = 0
        self.alpha = 0.95
        self.tao = 0.1
        self.cp = 2
        self.T = math.ceil(N/problem.n_var) # the number of neighbour vector
        # self.w_lambda_vector = uniform_vector_nbi(self.N, problem.n_obj)
        self.w_lambda_vector = get_reference_directions("das-dennis", self.n_objs, n_points=self.N)# [0:self.N]
        self.neighbour_w_ind = self.get_neighbour_vector(self.w_lambda_vector)
        self.ideal_point = np.zeros((self.last_gen, problem.n_obj))# store kth gen ideal point and k-1th ideal point
        self.nadir_point = np.zeros((self.last_gen, problem.n_obj))# store kth gen nadir point and k-1th nadir point
        self.pop = self.create_pop(self, N, problem.n_var, problem.n_obj, self.n_cv)
        self.crossover = tools.cxUniform
        self.mutation = tools.mutPolynomialBounded
        self.select = tools.selRandom
        # self.repair = tools.repair() 需要判断是否需要repair方法
        self.Z = np.zeros(self.n_objs)
        self.arch = []
        self.non_dominate_sort = NonDominatedSorting()
        self.calculate_crowdig_distance = calc_crowding_distance
        self.problem_pf = self.problem.pareto_front()
        self.mutation_func = get_mutation("real_pm", eta=self.mutation_eta, prob=1.0 / self.n_var)

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
        CR = 0.5
        offspring_chromosome = parents[0]
        rand_site = np.random.random((1, self.n_var))>CR
        offspring_chromosome_rep = parents[0] + 0.5 * (parents[1] - parents[2])
        offspring_chromosome = np.where(np.logical_and(offspring_chromosome, rand_site), offspring_chromosome_rep, offspring_chromosome)
        # offspring_chromosome = parents[0] + 0.5 * (parents[1] - parents[2])
        test_off = offspring_chromosome
        # offspring_chromosome = self.mutation(offspring_chromosome, eta=self.mutation_eta,
        #                                      low=self.var_low_bound, up=self.var_up_bound, indpb=1.0 / self.n_var)

        offspring_chromosome = mutation(self.mutation_func, np.array(offspring_chromosome))
        offspring = Individual(offspring_chromosome[0], self.n_var, self.n_cv)
        if offspring.shape != (self.n_var, ):
            raise Exception("offspring illegal!", offspring, test_off)

        off_evaluate_result = self.problem.evaluate(offspring, return_as_dictionary=True)
        offspring.fitness = off_evaluate_result['F']

        if 'G' in off_evaluate_result:
            offspring.cv = off_evaluate_result['G']

        return offspring

    def get_pop_cv_matrix(self):
        cv_matrix = np.zeros((self.N, self.n_cv))
        for i, ind in enumerate(self.pop):
            cv_matrix[i] = ind.cv

        return cv_matrix

    @staticmethod
    def calculate_overall_cv(cv_matrix):
        if len(cv_matrix.shape) != 2:
            cv_matrix = np.array([cv_matrix])

        cv_matrix = np.where(cv_matrix <= 0, 0, cv_matrix)
        cv_matrix = abs(cv_matrix)

        overall_cv = np.sum(cv_matrix, axis=1)
        return overall_cv

    def evaluate(self):
        evaluate_result_dic = self.problem.evaluate(np.array(self.pop), return_as_dictionary=True)
        evaluate_objs = evaluate_result_dic['F']
        # pf = self.problem.pareto_front()
        # Scatter(legend=True).add(pf, label="Pareto-front").add(evaluate_result_dic['F'], label="Result").show()

        for index, ind_obj in enumerate(evaluate_objs):
            self.pop[index].fitness = ind_obj

        if 'G' in evaluate_result_dic:
            evaluate_cvs = evaluate_result_dic['G']
            for index, ind_cvs in enumerate(evaluate_cvs):
                # print("inds cv lin 147", ind_cvs)
                self.pop[index].cv = ind_cvs


    def get_fitness_matrix(self):
        fitness_matrix = np.zeros((self.N, self.n_objs))
        for index, ind in enumerate(self.pop):
            fitness_matrix[index] = ind.fitness

        return fitness_matrix

    def get_pop_all_message_matrix(self, overall_cv_vector):
        decs_matrix = np.array(self.pop)
        objs_matrix = self.get_fitness_matrix()
        cv_matrix = np.array([overall_cv_vector]).T
        pop_obj_cv_mat = np.hstack((decs_matrix, objs_matrix, cv_matrix))
        return pop_obj_cv_mat

    def update_ideal_point(self, gen):
        X = self.get_fitness_matrix()
        now_ideal_point = np.min(self.get_fitness_matrix(), axis=0)
        if gen > self.last_gen-1:
            temp = list(self.ideal_point)
            temp.pop(0)
            temp.append(now_ideal_point)
            self.ideal_point = np.array(temp)
        else:
            self.ideal_point[gen] = now_ideal_point
        logger.debug(f"ideal_point: {self.ideal_point}")
        # self.ideal_point[0] = self.ideal_point[1]
        # self.ideal_point[1] = now_ideal_point
        # logger.debug(f"now ideal_point: {self.ideal_point[1]}")

    def update_nadir_point(self, gen):
        now_nadir_point = np.max(self.get_fitness_matrix(), axis=0)
        if gen > self.last_gen-1:
            temp = list(self.nadir_point)
            temp.pop(0)
            temp.append(now_nadir_point)
            self.nadir_point = np.array(temp)
        else:
            self.nadir_point[gen] = now_nadir_point
        # self.nadir_point[0] = self.nadir_point[1]
        # self.nadir_point[1] = now_nadir_point
        # logger.debug(f"now nadir_point: {self.nadir_point[1]}")
        logger.debug(f"nadir_point: {self.nadir_point}")

    def calc_max_change(self):
        delta_value = 1e-6 * np.ones(self.n_objs)
        abs_ideal_point_k_1 = np.abs(self.ideal_point[0])
        abs_nadir_point_k_1 = np.abs(self.nadir_point[0])
        # delta_value = 1e-6
        rz = np.max(np.abs(self.ideal_point[-1] - self.ideal_point[0]) / np.where(abs_ideal_point_k_1 > delta_value,
                                                                                 abs_ideal_point_k_1, delta_value))
        rnk = np.max(np.abs(self.nadir_point[-1] - self.nadir_point[0]) / np.where(abs_nadir_point_k_1 > delta_value,
                                                                                 abs_nadir_point_k_1, delta_value))
        # print("[rz, rnk]: [%f, %f]", rz, rnk)

        logger.info(f"rz = {rz}, rnk = {rnk}")
        return max(rz, rnk)

    def update_epsilon(self, epsilon_k, epsilon_0, rf, gen):
        if rf < self.alpha:
            result = (1 - self.tao) * epsilon_k

        else:
            result = epsilon_0 * (1 - (gen / self.Tc))**self.cp

        return result

    def push_sub_problem(self, P, offspring, g_old, g_new):
        nr_index = P[np.where(g_old > g_new)][0:self.nr]
        if nr_index.shape[0] >= self.nr:
            for i in range(self.nr):
                self.pop[nr_index[i]] = offspring

        elif nr_index.shape[0] == 1:
            self.pop[nr_index[0]] = offspring

        else:
            return

    def pull_sub_problem(self, P, offspring, g_old, g_off, cv_old, cv_off):
        g_can_replace = (g_old >= g_off)
        cv_can_replace_1 = np.logical_and((cv_old <= self.epsilon_k), cv_off <= self.epsilon_k)
        cv_can_replace_2 = (cv_old == cv_off)
        cv_can_replace_3 = (cv_off < cv_old)
        # replace_index1 = np.where(np.logical_and(g_can_replace, np.logical_or(np.logical_or(cv_can_replace_1,
        #                                                                                    cv_can_replace_2),
        #                                                                                       cv_can_replace_3)))
        replace_index1 = np.where(g_can_replace & (cv_can_replace_1 | cv_can_replace_2) | cv_can_replace_3)
        replace_index2 = replace_index1[0]
        # logger.debug(f"cv_can_replace_1: {cv_can_replace_1}")
        replace_index = np.random.permutation(replace_index2)[0:self.nr]
        # logger.debug(f"cv_can_replace_1: {cv_can_replace_1}")
        # logging.debug(f"replace index :{replace_index}")
        # logging.debug(f"offspring:{offspring}")
        # logger.info(f"replace index: {replace_index}")
        if replace_index.shape[0]:
            for re_index in P[replace_index]:
                self.pop[re_index] = offspring
        # for u in replace_index:
        #     if u.size:
        #         for i in u:
        #             self.pop[i] = offspring

    @staticmethod
    def chebyshev_decomposition_calculate(pop_objs, weights, z):
        ideal_point_temp = np.repeat(np.array([z]), pop_objs.shape[0], axis=0)
        g_chebyshev = np.max((pop_objs-ideal_point_temp)*weights, axis=1)
        return g_chebyshev

    def transform_pop_to_matrix(self, any_pop: list):

        chromosome_mat = np.zeros((len(any_pop), self.n_var))
        for index, individual in enumerate(any_pop):
            chromosome_mat[index] = individual

        objs_mat = np.zeros((len(any_pop), self.n_objs))

        # print("try fitness", any_pop[0].fitness)

        for index, individual in enumerate(any_pop):
            objs_mat[index] = individual.fitness

        cvs_mat = np.zeros((len(any_pop), self.n_cv))
        for index, individual in enumerate(any_pop):
            cvs_mat[index] = individual.cv

        return np.hstack((chromosome_mat, objs_mat, cvs_mat))


    def non_dominate_select(self):

        logger.debug("use arch select")
        temp_pop_arch = copy.deepcopy(self.pop)
        # print("temp fitness", temp_pop_arch[9].fitness)
        for i in range(len(self.arch)):
            temp_pop_arch.append(self.arch[i])

        pop_arch_mat = self.transform_pop_to_matrix(temp_pop_arch)
        constrains_mat = pop_arch_mat[:, self.n_var+self.n_objs:self.n_var+self.n_objs+self.n_cv]
        feasible_index = np.where(np.all(constrains_mat<=0, axis=1))[0]
        # logger.debug(f"feasible_index.shape[0]: {feasible_index.shape[0]}")
        if feasible_index.shape[0] > 0:
            arch_pop = pop_arch_mat[feasible_index]
            non_fronts_rank = self.non_dominate_sort.do(arch_pop[:, self.n_var:self.n_var+self.n_objs])
            # non_fronts is array
            non_fronts = non_fronts_rank[0]

            logger.info(f"non_fronts number is {non_fronts.shape[0]}")
            # logger.debug(f"non fronts ind is {non_fronts}")
            arch_pop = arch_pop[non_fronts]
            self.arch = []
            if non_fronts.shape[0] > self.N:
                crowing_distance = self.calculate_crowdig_distance(arch_pop[:, self.n_var:self.n_var+self.n_objs])
                arch_index = np.argsort(crowing_distance)[::-1][:self.N]
                arch_pop = arch_pop[arch_index]

                for index in range(arch_pop.shape[0]):
                    arch_ind = Individual(arch_pop[index][:self.n_var], self.n_objs, self.n_cv)
                    arch_ind.fitness = arch_pop[index][self.n_var:self.n_var+self.n_objs]
                    arch_ind.cv = arch_pop[index][self.n_var+self.n_objs:]
                    if arch_ind.cv.shape != (self.n_cv, ):
                        raise Exception("cv shape illegal!", arch_pop[self.n_var+self.n_objs:])
                    self.arch.append(arch_ind)

            else:
                for index in range(arch_pop.shape[0]):
                    arch_ind = Individual(arch_pop[index][:self.n_var], self.n_objs, self.n_cv)
                    arch_ind.fitness = arch_pop[index][self.n_var:self.n_var + self.n_objs]
                    arch_ind.cv = arch_pop[index][self.n_var + self.n_objs:]
                    if arch_ind.cv.shape != (self.n_cv, ):
                        raise Exception("cv shape illegal!", arch_pop[self.n_var+self.n_objs:])
                    self.arch.append(arch_ind)
        # print(f"arch num: {len(self.arch)}")
        # logger.debug(f"arch num: {len(self.arch)}")

    def draw(self, gen):

        plot = Scatter(title=str(gen))
        plot.add(self.problem.evaluate(np.array(self.pop), return_as_dictionary=True)['F'], s=30, facecolors='none',
                  edgecolors='red')
        print("arch num:", len(self.arch))
        if len(self.arch) != 0:
            plot.add(self.problem.evaluate(np.array(self.arch), return_as_dictionary=True)['F'], s=30, facecolors='blue',
                     edgecolors='blue')
        plot.add(self.problem_pf, s=30, facecolors="none", edgecolors="green")
        plot.show()

    def evolution(self):
        max_change = 10000
        self.evaluate()
        self.Z = np.min(self.get_fitness_matrix(), axis=0)
        for gen in range(self.max_gen):
            if gen == 100:
                print("debug")
            print("gen: ", gen)
            # self.problem.evaluate(np.array(self.pop))
            self.evaluate()
            pop_cvs = self.get_pop_cv_matrix()
            overall_cv_vector = self.calculate_overall_cv(pop_cvs)
            population = self.get_pop_all_message_matrix(overall_cv_vector)
            rf = np.sum(overall_cv_vector <= 1e-6) / self.N
            logger.debug(f"rf is {rf}")
            self.update_ideal_point(gen)
            self.update_nadir_point(gen)

            if gen >= self.last_gen:
                max_change = self.calc_max_change()
                logger.debug(f"max change: {max_change}")

            logger.info(f"Tc: {self.Tc}")
            logger.info(f"gen: {gen}")

            if gen < self.Tc:
                if max_change < self.change_threshold and self.search_stage == 1:
                    self.search_stage = -1
                    self.epsilon_0 = np.max(population[:, -1])
                    self.epsilon_k = self.epsilon_0
                    logger.debug(f"max change: {max_change}")

                if self.search_stage == -1:
                    self.epsilon_k = self.update_epsilon(self.epsilon_k, self.epsilon_0, rf, gen)
                    logger.info(f"use pull stage, gen is {gen}")
                    logger.debug(f"epsilon_k is {self.epsilon_k}")
            else:
                self.epsilon_k = 0

            for ind_index in range(self.N):

                if np.random.random() < self.delta:
                    # print("P wind")
                    P = np.random.permutation(self.neighbour_w_ind[ind_index, :])
                else:
                    # print("P N")
                    P = np.random.permutation(np.arange(self.N))

                offspring = self.differential_evolution_operator([self.pop[ind_index], self.pop[P[0]], self.pop[P[1]]])
                self.Z = np.where(self.Z < offspring.fitness, self.Z, offspring.fitness)
                # print("P", P)
                # print("fitness mat", self.get_fitness_matrix().shape)
                g_old = self.chebyshev_decomposition_calculate(self.get_fitness_matrix()[P], self.w_lambda_vector[P],
                                                               self.Z)

                g_off = self.chebyshev_decomposition_calculate(np.repeat(np.array([offspring.fitness]), P.shape[0],
                                                                         axis=0), self.w_lambda_vector[P], self.Z)

                cv_old = self.calculate_overall_cv(self.get_pop_cv_matrix()[P])

                cv_off = self.calculate_overall_cv(offspring.cv*np.ones((P.shape[0], 1)))

                if self.search_stage == 1:
                    # logger.info(f"use push stage, gen is {gen}")
                    self.push_sub_problem(P, offspring, g_old, g_off)

                else:
                    # logger.info(f"use pull stage, gen is {gen}")
                    self.pull_sub_problem(P, offspring, g_old, g_off, cv_old, cv_off)

            self.non_dominate_select()
            # self.draw(gen)
            # if gen % 400 and saveData:
            #     print("save")
            #     save_data(f"data_LIORCMOP1_{gen}", self.pop, self.arch)
        self.draw(self.max_gen)
        # save_data(f"Cdtlz_{self.max_gen}", self.pop, self.arch)
        return self.arch, self.pop


def save_data(file_name, pop, arch):
    path = sys.path[0]
    path = os.path.join(path, "data")
    path = path + "/" + file_name + ".npz"
    np.savez(path, pop=pop, arch=arch)

if __name__ == '__main__':
    # my_problem = C1_DTLZ1()
    my_problem = get_problem("ctp1")
    # my_problem = LIRCMOP1()
    # arc_mat = np.zeros((len(arch), my_problem.n_var))
    # arc_mat = np.zeros((10, 10))
    # save_data("test", arc_mat, arc_mat)
    moeadpps_object = MoeadPPS(91, my_problem, 30000)
    # print("nei", moeadpps_object.w_lambda_vector.shape)

    arch, pop = moeadpps_object.evolution()
    arch_np = np.array(arch)

    # arc_mat = np.zeros((len(arch), my_problem.n_var))
    # pop_mat = np.zeros((len(pop), my_problem.n_var))
    # for i in range(len(arch)):
    #     arc_mat[i] = arch[i]
    # for i in range(len(pop)):
    #     pop_mat[i] = pop[i]
    #  res = my_problem.evaluate(arc_mat, return_as_dictionary=True)
    # # print(res)
    # # get_visualization("scatter").add(res['F']).show()
    # print(res)
    #
    # res_pop = my_problem.evaluate(pop_mat, return_as_dictionary=True)
    # print(res_pop)
    # pf = my_problem.pareto_front()
    # Scatter(legend=True).add(pf, label="Pareto-front").add(res_pop['F'], label="Result").show()
    # print(arch)
    #
    # print(len(arch))
    # print(res)





