from moeadpps.moeadpps import MoeadPPS
from pymoo.factory import get_problem
import numpy as np

def evolve():
    my_problem = get_problem("ctp1")
    moeadpps_object = MoeadPPS(91, my_problem, 30)
    arch, pop = moeadpps_object.evolution()
    arch_np = np.array(arch)


if __name__ == '__main__':
    evolve()







