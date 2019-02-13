import numpy as np

a = np.array([[0,1,2],[3,4,5],[6,7,8]])
print(a)

mins = a.min(axis=0)
print(mins)

for col in range(a.shape[1]):
	a.T[col] = a.T[col] - mins[2]
print(a)
