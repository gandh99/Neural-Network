import numpy as np
import matplotlib.pyplot as plt

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

# x=[1,2,3,4,5]
# y1=[10,11,12,13,15]
# y2=[20,21,22,23,24]

# plt.plot(x, y1, marker="x", label="y1")
# plt.plot(x, y2, marker="x", label="y2")


# plt.xlim(1,50)
# plt.ylim(1,50)

# plt.xlabel('Item (s)')
# plt.ylabel('Accuracy')
# plt.title('Python Line Chart: Plotting numbers')
# plt.grid(True)
# plt.legend()
# plt.show()

newRow = np.zeros((1, 4))
newRow[:, 3] = 1
print(newRow)