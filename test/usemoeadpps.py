from pymoo.factory import get_problem

from moeadpps.moeadpps import MoeadPPS


my_problem = get_problem("dascmop1", 1)
algorithm = MoeadPPS(300, my_problem, 300*2000)

arch, pop = algorithm.evolution()
algorithm.draw(2000)

print(arch)
print(pop)

# from pymoo.factory import get_problem
# from pymoo.util.plotting import plot
#
# problem = get_problem("dascmop1", 1)
# plot(problem.pareto_front(), no_fill=True)


