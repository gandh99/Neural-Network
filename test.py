import numpy as np

a = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
b = np.array([[0, 0, 0, 1], [1, 0, 0, 0]])
# print((a == b).all())

# diff = abs(a - b)
# summed = np.sum(diff, axis=1)
# unique, counts = np.unique(summed, return_counts=True)
# ans = dict(zip(unique, counts))
# print(diff)
# print(summed)
# print(ans)
# print(type(ans))

# ans = np.transpose(np.nonzero(a))
# ans = ans[:, 1]
# print(ans)

# ans = np.argmax(a, axis=1)
# print(ans)

dic = {1:1, 2:2, 4:10}
print(len(dic))