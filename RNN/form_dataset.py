import os
import numpy as np
import pickle
import random
from tqdm import tqdm

data = []
training_data = []
testing_data = []

train_split = .8

groups = [[1, 2, 3], [11, 13, 14, 26, 27, 28], [9, 24, 31, 33], [6, 7, 8, 20, 25, 21, 23], [17, 19, 29, 32], [10, 12], [18, 22], [4, 5, 15, 16, 30]]
group_data = [[] for i in groups]

dir = 'raw_trajectories/all_trajs/'

for _, _, files in os.walk(dir):
    for file in tqdm(files):
        trial_data = np.load(dir + file)
        grasp_num = int(file.split('_')[1])
        group_num = np.where([grasp_num in g for g in groups])[0][0]
        data.append(trial_data)
        group_data[group_num].append(trial_data)

train_trials = random.sample(list(np.arange(len(data))), np.floor(.6*len(data)).astype(int))

for i, d in tqdm(enumerate(data)):
    if i in train_trials:
        training_data.append(d)
    else:
        testing_data.append(d)

data = np.concatenate(data, axis = 0)
training_data = np.concatenate(training_data, axis = 0)
with open('test_data_seperate.pickle', 'wb') as output_file:
    pickle.dump(testing_data, output_file)
testing_data = np.concatenate(testing_data, axis = 0)

np.save('dataset_data', data)
np.save('training_data', training_data)
np.save('testing_data', testing_data)

group_data = [np.concatenate(g) for g in group_data]
for i, g in enumerate(group_data):
    np.save('group_data/{}_data'.format(i+1), g)
