import numpy as np
from problem.LIRCMOP.lircmop1 import LIRCMOP1
from pymoo.factory import get_problem
from pymoo.visualization.scatter import Scatter
import scipy.io

path = "/Users/wheat/CODE/MyCodeProject/MOEAD_PPS/moeadpps/data/data_ctp1.npz"
lir_path = "/Users/wheat/CODE/MyCodeProject/MOEAD_PPS/moeadpps/data/data.npz"
matlab_data_path = "/Users/wheat/CODE/MyCodeProject/MOEAD_PPS/moeadpps/data/good_res.mat"
data = np.load(path)
pop = data['pop']
arch = data['arch']

# print(pop)
# print(arch)

al_problem = LIRCMOP1()
# ctp_pro = get_problem("ctp1")
# res_pop = ctp_pro.evaluate(pop, return_as_dictionary=True)
# res_arch = ctp_pro.evaluate(arch, return_as_dictionary=True)
# print(res_pop)
# print(res_arch)

# lirdata = np.load(lir_path)
# lir_pop = lirdata['pop']
# lir_arch = lirdata['arch']
# res_lir_pop = al_problem.evaluate(lir_pop, return_as_dictionary=True)
# res_lir_arch = al_problem.evaluate(lir_arch, return_as_dictionary=True)

# print(" ")


def draw(pro_name, problem, pops, archs):
    plot = Scatter(title=pro_name)
    plot.add(problem.evaluate(pops, return_as_dictionary=True)['F'], s=30, facecolors='none',
             edgecolors='red')
    print("arch num:", archs.shape[0])
    plot.add(problem.evaluate(archs, return_as_dictionary=True)['F'], s=30, facecolors='blue',
             edgecolors='blue')
    plot.add(problem.pareto_front(), s=10, facecolors="none", edgecolors="green")
    plot.show()


# draw("ctp1", ctp_pro, pop, arch)
# draw("LIRCMOP1", al_problem, lir_pop, lir_arch)
mat = scipy.io.loadmat(matlab_data_path)
# print(mat['a'])
mat_pops = mat['pops']
mat_archs = mat['archs']
draw("LIRCMOP1 mat", al_problem, mat_pops, mat_archs)
res_mat = al_problem.evaluate(mat_pops, return_as_dictionary=True)
res_arch_mat = al_problem.evaluate(mat_archs, return_as_dictionary=True)
print(' ')

