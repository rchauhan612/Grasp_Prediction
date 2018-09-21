import numpy as np
import os

files = os.listdir('results/expanded')
data = np.zeros((1, 15))
for file in files:
    temp = np.load('results/expanded/' + file)
    temp = np.array([temp[-1, :]])
    temp *= np.pi/180.
    data = np.append(data, temp, axis = 0)

data = np.delete(data, 0, axis = 0)
data = np.delete(data, 0, axis = 1)


n = data.shape[0]
avg = np.mean(data, axis = 0)
X = data - np.repeat([avg], n, axis = 0)
C = np.matmul(X.T, X)
weight, PC = np.linalg.eig(C)
weight = np.real(weight)
PC = np.real(PC)
weight = weight / sum(weight)

np.save('PCAs/Glove_PCA_weights', weight)
np.save('PCAs/Glove_PCA', PC)
