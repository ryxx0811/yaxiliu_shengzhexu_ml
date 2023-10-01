import numpy as np
import random
from collections import namedtuple

ACTIONS=['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
weights = np.random.rand(len(ACTIONS))
model = weights / weights.sum()
print(model)