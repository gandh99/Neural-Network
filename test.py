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

x=[1,2,3,4,5]
y=[10,11,12,13,15]

plt.plot(x, y, marker="x")

plt.xlim(1,50)
plt.ylim(1,50)

plt.xlabel('Item (s)')
plt.ylabel('Accuracy')
plt.title('Python Line Chart: Plotting numbers')
plt.grid(True)
plt.show()