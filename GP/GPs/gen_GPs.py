import numpy as np
from scipy import signal
import os
import matplotlib.pyplot as plt

traj_loc = '../Trajectories/processed/scaled/'

for i in range(1, 31): #(1, 31)
    file_list = os.listdir(traj_loc+'Subject '+str(i))
    if not os.path.exists('Subject ' + str(i)):
        os.makedirs('Subject ' + str(i))
    for j in range(1, 34): #(1, 34)
        data = []
        for trial_name in file_list:
            if (int(trial_name.split(' ')[0]) == j):
                data.append(np.load(traj_loc + 'Subject ' + str(i) + '/' + trial_name))
        trial_size = data[0].shape
        grasp_mean = np.zeros(trial_size)
        grasp_var = np.zeros(trial_size)
        for k in range(len(data[0])):
            time_step = data[0][k, :]
            for w in range(1, len(data)):
                time_step = np.vstack((time_step, data[w][k, :]))
            grasp_mean[k, :] = np.mean(time_step, axis = 0)
            grasp_var[k, :] = np.var(time_step, axis = 0)
        np.save('Subject ' + str(i) + '/Subject ' + str(i) + '_g' + str(j) + '_mean', grasp_mean)
        np.save('Subject ' + str(i) + '/Subject ' + str(i) + '_g' + str(j) + '_var', grasp_var)
        # print(grasp_mean)
        # for z in range(6):
        #     plt.plot(np.arange(20), grasp_mean[:, z])
        # plt.show()
