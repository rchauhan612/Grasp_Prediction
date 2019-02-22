import numpy as np
import matplotlib.pyplot as plt
import sys

filename = input('file name:\n')
if len(sys.argv) > 0:
    if 'exp' in sys.argv:
        print('getting expanded data')
        data = np.load('results/expanded/'+ filename+'_expanded.npy')
    else:
        data = np.load('results/'+ filename+'.npy')
else:
    data = np.load('results/'+ filename+'.npy')
print(data.shape)
for i in range(1, len(data[0])):
    plt.plot(data[:, 0], data[:, i])

# plt.legend([str(i) for i in range(len(data[0]) - 2)])
plt.show();
