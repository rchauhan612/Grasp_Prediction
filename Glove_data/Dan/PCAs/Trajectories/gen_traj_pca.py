import numpy as np
import os
import sys
import matplotlib.pyplot as plt

colors = plt.cm.get_cmap('tab10')(np.linspace(0,0.6,6))

if 'gen' in sys.argv:
    files = os.listdir('../../results/expanded/')
    PC = np.load('../Glove_PCA.npy')[:, :6]

    for file in files:
        data = np.load('../../results/expanded/' + file)
        data2 = np.zeros((len(data), 6+1))
        for i in range(len(data)):
            data2[i, 1:] = np.matmul(np.array([data[i, 1:]]), PC)
            data2[i, 0] = data[i, 0]
        np.save(file.split('.')[0]+'_PCA', data2)

elif 'plot' in sys.argv:
    files = os.listdir()
    for file in files:
        if not file == 'gen_traj_pca.py':
            fig = plt.figure()
            data = np.load(file)
            for i in range(5):
                plt.plot(data[:, 0], data[:, i+1], color = colors[i])

            plt.title(file)
            plt.show()
            plt.close(fig)
else:
    print('enter gen or plot')
