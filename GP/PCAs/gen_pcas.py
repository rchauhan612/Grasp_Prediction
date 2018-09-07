#! /usr/bin/python

import numpy as np
import math
import os

d = 16
q = 6
trajs = []

for root0, files0, names0 in os.walk('../../HUSTdataset/Subjects', topdown = True):
    cnt0 = 0
    for file0 in files0:
        if (os.path.exists(os.path.join('../../HUSTdataset/Subjects', file0))):
            data = np.zeros((1, d))
            print(file0)
            cnt0 += 1
            subject_path = '../../HUSTdataset/Subjects' + '/' + file0
            for root, files, names in os.walk(subject_path, topdown = True):
                cnt = 0
                for file in files:
                    if (os.path.exists(os.path.join(subject_path, file))):
                        cnt+=1
                        for root1, files1, names1 in os.walk(subject_path + '/' + file, topdown = True):
                            cnt2 = 0
                            grasp_trajs = []
                            for file1 in files1:
                                cnt2+=1
                                for name1 in range(1, 4):
                                    r = np.genfromtxt(subject_path + '/' + file + '/' + file1 + '/' + str(name1) + '.txt', delimiter='\t')
                                    grasp_trajs.append(r)
                                    data = np.append(data, [r[r.shape[0]-1, 1:d+1]], axis=0) # only end configuration
                            if (len(grasp_trajs) > 0):
                                trajs.append(grasp_trajs)

            data = np.delete(data, 0, axis = 0)
            data = np.delete(data, [0, 1], axis = 1)
            n = data.shape[0]
            avg = np.mean(data, axis = 0)
            X = data - np.repeat([avg], n, axis = 0)
            C = np.matmul(X.T, X)
            weight, PC = np.linalg.eig(C)
            weight = weight / sum(weight)
            np.save('results/'+file0.split(' ')[0]+'_'+file0.split(' ')[1]+'_PCA', PC)
