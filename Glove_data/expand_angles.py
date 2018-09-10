import numpy as np
import os

files = os.listdir('results')
temp = np.zeros(15)
coupling = [1, 0.98, .87]
for file in files:
    if (len(file.split('.')) > 1):
        angles = [np.zeros(15)]
        data = np.load('results/'+file)
        for i in range(len(data)):
            temp2 = -1*(data[i, 1:] - 480)*45/100
            # print(data[i, 1:])
            temp[0] = data[i, 0]
            for j in range(1, 5):
                for w in range(0, 3):
                    temp[1+w+(j-1)*3] = temp2[j] * coupling[w]
            temp[13] = temp2[5]
            temp[14] = temp2[5] * 1.1
            # print(temp)
            angles = np.append(angles, [temp], axis = 0)
        np.save('results/expanded/'+file.split('.')[0]+'_expanded', angles)
