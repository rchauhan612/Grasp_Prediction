import numpy as np
import os
import re
import sys
sys.path.append('../../GP/')
from gaussian_process import GP
import pickle
import matplotlib.pyplot as plt

def predict_trial(gp_names, gp_list, weight, trial_data):
    (l, q) = trial_data.shape
    # q -= 1
    q = 5
    weight = weight[:q]
    time = trial_data[:, 0]
    time = 65*time / time[-1] # remove this when time scaling gets implemented
    traj = np.pi*trial_data[:, 1:q+1]/180. #the pca was done in deg, moving to radians
    for i in range(1, int(.5*l)):
        grasp_llhs = np.zeros(len(grasp_gps))
        for j in range(len(grasp_gps)):
            grasp_gp = grasp_gps[j]
            grasp_llhs[j] = 0
            for pc in range(q):
                # print(grasp_gp[pc])
                # print(grasp_gp[pc].sample_values_y[0])
                grasp_llhs[j] += weight[pc] * grasp_gp[pc].calc_likelihood(1, 0, time[:i], traj[:i, pc], False, 5)
                # print(j, pc, grasp_gp[pc].calc_likelihood(1, 0, time[:i], traj[:i, pc], False, 5))

    # print(grasp_llhs, gp_names[np.argmax(grasp_llhs)])
    return(gp_names[np.argmax(grasp_llhs)])
        # print(time[i], traj[i])

trial_dir = '../PCAs/Trajectories/'

trial_data_loc_list= []
trial_data_list = []
trial_names = []

for file in os.listdir(trial_dir):
    if file.split('.')[1] == 'npy':
        trial_data_loc_list.append(trial_dir + file)
        trial_data_list.append(np.load(trial_data_loc_list[-1])[:96, :])
        trial_names.append(file)
        # trial_names.append(file.split('_')[0])

gp_file = open('glove_GPs', 'rb')
gp_data = pickle.load(gp_file)
grasp_gps = gp_data[1]
# print(len(grasp_gps))
# print(grasp_gps)
grasp_names = gp_data[0]
print(grasp_names)

# weight = np.load('../PCAs/Glove_PCA_weights.npy')
weight = np.ones(5) * .2
acc= 0
cnt = 0
for i in range(len(trial_data_list)):
    cnt +=1
    # if i == 1:
    #     break
    pred = predict_trial(grasp_names, grasp_gps, weight, trial_data_list[i][1:, :]) #the first row of the array appears to be all zeros, i should have fixed this in the generation but im just doing it here
    # corr = trial_names[i][:re.search('\d', trial_names[i]).start()] == pred
    corr = trial_names[i].split('_')[0] == pred
    acc += int(corr)
    if not False:
        print(trial_names[i].split('_')[0], '\t', pred, '\t', corr)

print(100*acc/cnt)
