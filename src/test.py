import numpy as np
A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]

# print(A)
# print(B)


C = np.vstack((A,B))
print(C)
