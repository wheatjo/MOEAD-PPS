import numpy as np
from pyrecorder.recorder import Recorder
from pyrecorder.writers.video import Video

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter

for k in range(10):
    X = np.random.random((100, 2))
    Scatter(title=str(k)).add(X).show()