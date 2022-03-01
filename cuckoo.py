from random import uniform
from random import randint 
import math

import numpy as np
from utils import minmin, get_fitness

array = np.array(open('512x16/u_c_hihi.0').readlines(),dtype=float)
ET = np.reshape(array,(512, 16))
CT = ET.copy()

print(ET)

m, i = minmin(ET, CT, np.zeros(16,dtype=float))

