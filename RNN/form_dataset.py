import os
import numpy as np
import pickle
import random

data = []
training_data = []
testing_data = []

train_split = .6

dir = 'raw_trajectories/all_trajs/'

for _, _, files in os.walk(dir):
    for file in files:
        trial_data = np.load(dir + file)
        data.append(trial_data)

train_trials = random.sample(list(np.arange(len(data))), np.floor(.6*len(data)).astype(int))

for i, d in enumerate(data):
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
