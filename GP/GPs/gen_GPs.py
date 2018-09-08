import numpy as np
from scipy import signal
import os
import sys
sys.path.append('..')
from gaussian_process import GP
import matplotlib.pyplot as plt
import pickle

traj_loc = '../Trajectories/processed/scaled/'
colors = plt.cm.get_cmap('tab10')(np.linspace(0,0.6,6))

for i in range(1, 31): #(1, 31)
    file_list = os.listdir(traj_loc+'Subject '+str(i))
    if not os.path.exists('Subject ' + str(i)):
        os.makedirs('Subject ' + str(i))
    subject_gps = []
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
        grasp_gps = []
        # plt.figure()
        # ax = plt.gca()
        for k in range(trial_size[1]):
            gp = GP(np.arange(0, trial_size[0]), grasp_var[:, k])
            gp.opt(grasp_mean[:, k])
            gp.eval_continuous(100)
            grasp_gps.append(gp)
            # gp.plot_process(ax, colors[k])
        subject_gps.append(grasp_gps)
        print('%.3f %%' % (100*((i-1)*33 + j) / (30*33)))
        # plt.show()
    outfile = open('Subject ' + str(i) + '/Subject ' + str(i) + '_GPs', 'wb')
    pickle.dump(subject_gps, outfile)
    outfile.close()
