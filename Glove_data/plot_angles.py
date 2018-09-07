import numpy as np
import matplotlib.pyplot as plt

filename = input('file name:\n')
data = np.load('results/'+ filename+'.npy')

for i in range(len(data[0]) - 2):
    plt.scatter(data[:, 0], data[:, i+1], s = 7)

plt.legend([str(i) for i in range(len(data[0]) - 2)])
plt.show();
