import numpy as np

f = np.array([1,2,3])
g = np.array([4,5,6])

print(f)
print(g)

h_stack = np.hstack((f,g))
print(h_stack)
v_stack = np.vstack((f,g))
print(v_stack)