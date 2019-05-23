import os
import numpy as np
import pickle

data = [[] for i in range(33)]

dir = 'raw_trajectories/all_trajs/'

max_trial_len = 0

for _, _, files in os.walk(dir):
    for file in files:
        trial_data = np.load(dir + file)
        max_trial_len = np.max((max_trial_len, trial_data.shape[0]))
        grasp_num = int(file.split('_')[1])
        data[grasp_num-1].append(trial_data)

labels = []

for i, grasp in enumerate(data):
    for j, trial in enumerate(grasp):
        trial_len = trial.shape[0]
        rep = np.ones(trial_len)
        rep[-1] = 1 + max_trial_len - trial_len
        grasp[j] = np.repeat(trial, repeats = rep.astype(int), axis = 0)
        labels.append(i+1)
    data[i] = np.array(grasp)

data = np.concatenate(data, axis = 0)

np.save('dataset_labels', labels)
np.save('dataset_data', data)
