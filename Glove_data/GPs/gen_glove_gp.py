import numpy as np
import os
import sys
sys.path.append('../../GP/')
from gaussian_process import GP
import pickle
import matplotlib.pyplot as plt

colors = plt.cm.get_cmap('tab10')(np.linspace(0,0.6,6))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

file_loc = '../PCAs/Trajectories/'
files = []
grasp_names = []
for file in os.listdir(file_loc):
    if file.split('.')[1] == 'npy':
        files.append(file)
        temp = file.split('_')[0]
        for i in range(len(temp)):
            if(not is_number(temp[i])):
                pass
            else:
                break
        grasp_names.append(temp[:i])

grasp_names = list(set(grasp_names))
grasp_data = [None] * len(grasp_names)
grasp_gps = [[None] * 5] * len(grasp_names)
trial_len = 40 #it just is
plt.figure(dpi = 60)
for i in range(len(grasp_names)):
    plt.subplot(5, 1, i+1)
    ax = plt.gca()
    temp = []
    for file in files:
        if grasp_names[i] in file:
            temp.append(np.load(file_loc+file)[:trial_len, :])
    grasp_data[i] = temp

    grasp_mean = np.zeros((trial_len, 5))
    grasp_var = np.zeros((trial_len, 5))
    for k in range(trial_len):
        time_step = np.pi*grasp_data[i][0][k, 1:-1]/180
        for w in range(1, len(temp)):
            time_step = np.vstack((time_step, np.pi*grasp_data[i][w][k, 1:-1]/180))
        grasp_mean[k, :] = np.mean(time_step, axis = 0)
        grasp_var[k, :] = np.var(time_step, axis = 0)
    for k in range(5):
        gp = GP(np.arange(0, trial_len), grasp_var[:, k])
        gp.opt(grasp_mean[:, k])
        gp.eval_continuous(45)
        grasp_gps[i][k] = gp
        gp.plot_process(ax, colors[k])
    plt.title(grasp_names[i])
    plt.xlim([0, 1])
    plt.xlabel('Grasp Progress(%)')
    plt.ylim([-2, 4])
    plt.ylabel('PC Magnitude')
outfile = open('glove_GPs', 'wb')
pickle.dump(grasp_gps, outfile)
outfile.close()
plt.show()
