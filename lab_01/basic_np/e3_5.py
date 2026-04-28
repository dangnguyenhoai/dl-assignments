import numpy as np

e = np.array([[1.,2.,3.],[4.,5.,6.],[4.,5.,6.],[4.,5.,6.]])

print(e)
print(e.shape)

f = e.reshape(2,3,2)
print(f)

g = f.flatten()
print(g)

