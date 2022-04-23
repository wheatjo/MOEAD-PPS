import numpy as np
import pymoo.core.population as pop
from pymoo.core.individual import Individual
# seed = np.random.seed(1)

a = np.random.random((10,2))

print(a)

pops = pop.Population(10)

print(pops)
i = pops[0]
print(i.__dict__.keys())

print(pops.get('X', 'F'))

u = pop.pop_from_array_or_individual(a)
u.set('F', a)
message = u.get('X', 'F')
print(message)
X = pops.new(10)
print(len(pops))
print(X.get('X'))

