from deap import base
from deap import tools
import numpy as np
import math
from utils.uniform_vector import uniform_vector_nbi

class MoeadPPS(object):

    def __init__(self, N, delta, nr, problem):
        self.N = N
        self.delta = delta
        self.nr = nr
        self.T = math.ceil(N/problem.n_var)
        self.w_lambda_vector = math.nch


