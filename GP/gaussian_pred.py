#! /usr/bin/python

import numpy as np
from numpy import (exp, absolute, pi, log, sqrt, mean)
from scipy import signal
from scipy.stats  import entropy
import math
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
# from plotly.offline import plot
# import plotly.graph_objs as go
import os
from hand import Finger, Thumb
from animator import get_frames
import time
from gaussian_process import GP
import pickle


def opt_gr_scale(scale0, w, GPs, x_eval, y_eval):
    # print('Start: %.3f' % (scale0))
    # print('-----')
    vals = []
    test_val = scale0*np.linspace(.1, 100, 50)
    for j in range(len(test_val)):
        vals.append(sum([w[i] * GPs[i].calc_likelihood(test_val[j], 0, x_eval, y_eval[i], False, 3) for i in range(len(w))]))
    # for i in range(len(test_val)):
    #     print(test_val[i], vals[i])
    # print(' ')
    return(test_val[np.argmax(vals)])
    # sol = minimize(lambda scale: -1*sum([w[i] * GPs[i].calc_likelihood(scale, 0, x_eval, y_eval[i], False, 3) for i in range(len(w))]),
    #     scale0, method = 'CG')#bounds=[(.66*scale0, 1.5*scale0)], options = {'maxiter': 100}
    # return(abs(sol.x))
    # return max(scale0, sol.x)

def eval_KL_div(p, q, b):
    # print(p, q)
    p = np.random.normal(p[0], p[1], 1000)
    q = np.random.normal(q[0], q[1], 1000)
    # print(entropy(p, q, b))
    return entropy(p, q, b)

# f = open('results.csv', 'w+')
d = 16
q = 6
trajs = []
test_subject = 1
PC = np.load('./16DOF/PCAs/Subject_' + str(test_subject) + '_PCA.npy')
PC = PC[:, 0:q]

sample_size = 20
# colors = cm.tab10(np.linspace(0,0.6,q))
colors = plt.cm.get_cmap('tab10')(np.linspace(0,0.6,33))
colors2 = plt.cm.get_cmap('prism_r')(np.linspace(0, .77,33))


#
# sim_grasp_num = test_grasp
# if (len(trajs[test_grasp - 1]) >= 3*(test_object-1) + test_trial):
#     test_data = trajs[test_grasp-1][3*(test_object-1) + test_trial - 1]
# else:
#     break

gp_file = open('./16DOF/GPs/Subject ' + str(test_subject) + '/Subject ' + str(test_subject) + '_GPs', 'rb')
GP_list = pickle.load(gp_file)

sim_grasp_num = 14
# test_name = '01 Large Diameter_Cylinder1_1_raw.npy'
#
# test_data = np.load('./16DOF/Trajectories/raw/Subject ' + str(test_subject) + '/' + test_name)
#
# dp = len(test_data)
# test_data_smoothed = np.zeros((1, sample_size))
#
# for i in range(0, q):
#     f_smooth = interp1d(np.linspace(0, 1, dp), test_data[:, i])
#     test_data_smoothed = np.append(test_data_smoothed, [f_smooth(test_x)], axis = 0)
# test_data_smoothed = np.delete(test_data_smoothed, 0, axis = 0)
# test_data_smoothed = test_data_smoothed.T

test_name = '14 Tripod_Sphere7_1_processed_scaled.npy'
test_data_smoothed = np.load('./16DOF/Trajectories/processed/scaled/Subject ' + str(test_subject) + '/' + test_name)

test_x = np.linspace(0, 1, sample_size)
test_scale = 1

test_x *=test_scale

test_GP = [GP(test_x[0:2]*sample_size, 0.1) for _ in range(q)]
test_pred_pt = [None for _ in range(q)]
# for i in range(q):
#     test_GP[i].opt(test_data_smoothed[0:2, i])

opt_scales = np.array([1. for _ in range(33)])

llhs = np.array([None for _ in range(33)])
llhs_text = [None for _ in range(33)]
llhs_pred = np.array([None for _ in range(33)])
llhs_pred_text = [None for _ in range(33)]
ents = np.array([0. for _ in range(33)])
ents_text = [None for _ in range(33)]

# print(sim_grasp)

prior_set =[[None for _ in range(q)] for _ in range(2)]
# proj_set =[[None for _ in range(q)] for _ in range(33)]
proj_set =[[None for _ in range(q)] for _ in range(2)]
most_ll_gp_set = [[None for _ in range(q)] for _ in range(2)]

opt_PC_cnt = 1


last_most_ll = sim_grasp_num
most_ll = sim_grasp_num
most_ll_hist = np.array([])
actual_scale_hist = np.array([])

ll_hist = np.zeros((1, 33))

weight = np.ones(q)

# print(test_data_smoothed)
plt_dat = np.zeros((1, 5))

for i in range(1, sample_size):
    j = 0
    for g in range(33):
        j+=1
        temp = 0
        temp2 = 0
        for k in range(0, q):
            plt.figure(num = 2)
            temp += weight[k] * GP_list[g][k].calc_likelihood(opt_scales[g], 0, sample_size*test_x[0:i], test_data_smoothed[0:i, k], False, 5)
        llhs[g] = temp
        llhs_pred[g] = temp2
        ent_sum = 0

    if(i > 1):
        # print(np.array([llhs]).shape)
        ll_hist = np.append(ll_hist, np.array([llhs]) / i, axis = 0)
        most_ll = 1 + np.argmax(llhs_pred + llhs)
        last_most_ll = most_ll
        most_ll_hist = np.append(most_ll_hist, most_ll)
        actual_scale_hist = np.append(actual_scale_hist, opt_scales[sim_grasp_num-1])
        five_most = np.argpartition(llhs_pred + llhs, -4)[-4:]
        print( most_ll, 1+five_most )# , (llhs_pred + llhs)[five_most]/i)
        # plt_dat = np.append(plt_dat, np.array([(llhs_pred + llhs)[five_most]/i]), axis = 0)
    # f.write(','.join(['%d' %_ for _ in test_gr]))
    # f.write(',')
    # f.write(','.join(['%d' %_ for _ in most_ll_hist]))
    # f.write('\n')
    data_formed = True
plt.ioff()
# plt.show()
# input('')
# plt.close('all')
# f.close()
print(five_most)
plt.subplot(2, 1, 1)
for i in range(4):
    plt.plot(np.arange(18), ll_hist[1:, five_most[i]], color = colors2[five_most[i]], label = 'Grasp ' + str(five_most[i]+1), lw = [1, 2.5][five_most[i] == sim_grasp_num-1])
plt.legend(ncol = 2)
plt.subplot(2, 1, 2)
for i in range(33):
    print([1, 5][i == sim_grasp_num-1])
    plt.plot(np.arange(18), ll_hist[1:, i], color = ['grey', colors2[i]][i == sim_grasp_num-1], lw = [1, 2.5][i == sim_grasp_num-1])
plt.show()
print('Done')
