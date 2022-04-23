from pymoo.algorithms.moo.moead import MOEAD, NeighborhoodSelection
from pymoo.operators.crossover.dex import DEX
from pymoo.operators.crossover.dex import de_differential
from pymoo.algorithms.soo.nonconvex.de import DES
import numpy as np


class MOEADPPSDE(MOEAD):

    def __init__(self, reference_point, variant="DE/rand/1/bin", CR=0.5, F=None, dither="vector",jitter=False):

        self.crossover = de_differential
        _, sel, n_diff, mut, = variant.split("/")
        selection = DES(sel)
        crossover = DEX(prob=1.0,
                        n_diffs=int(n_diff),
                        F=F,
                        CR=CR,
                        variant=mut,
                        dither=dither,
                        jitter=jitter)

        super().__init__(ref_dirs=reference_point, crossover=crossover, selection=selection)

    def _setup(self, problem, **kwargs):

        if isinstance(self.decomp, str):
            # for one or two objectives use tchebi otherwise pbi
            if self.decomp == 'auto':
                if self.problem.n_obj <= 2:
                    from pymoo.decomposition.tchebicheff import Tchebicheff
                    self.decomp = Tchebicheff()
                else:
                    from pymoo.decomposition.pbi import PBI
                    self.decomp = PBI()

    def _initialize_advance(self, infills=None, **kwargs):
        super()._initialize_advance(infills, **kwargs)
        self.ideal = np.min(self.pop.get("F"), axis=0)
        self.nadir = np.max(self.pop.get("F"), axis=0)

    def _advance(self, **kwargs):
        pop = self.pop
        for i in np.random.permutation(len(pop)):
            # get the parents using the neighborhood selection
            P = self.selection.do(pop, 1, self.mating.crossover.n_parents, k=[i])
            P = np.concatenate((P, np.array([i])))
            # perform a mating using the default operators - if more than one offspring just pick the first
            off = self.mating.do(self.problem, pop, 1, parents=P)[0]

            # evaluate the offspring
            self.evaluator.eval(self.problem, off, algorithm=self)

            # update the ideal point
            self.ideal = np.min(np.vstack([self.ideal, off.F]), axis=0)

            # now actually do the replacement of the individual is better
            self._replace(i, off)

    @staticmethod
    def calculate_overall_cv(pop):
        overall_cv = np.sum(pop.get("CV"), axis=1)
        return overall_cv

