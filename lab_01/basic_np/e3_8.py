import numpy as np

A = np.matrix(np.ones((5,5)))
print(A)

np.array(A)[2] = 2
print(A)
np.asarray(A)[2] = 3
print(A)