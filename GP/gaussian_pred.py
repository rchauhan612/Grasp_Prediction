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

f = open('results.csv', 'w+')
d = 16
q = 6
trajs = []
test_subject = 1
PC = np.load('./PCAs/Subject_' + str(test_subject) + '_PCA.npy')
PC = PC[:, 0:q]

sample_size = 20
# colors = cm.tab10(np.linspace(0,0.6,q))
colors = plt.cm.get_cmap('tab10')(np.linspace(0,0.6,q))

#
sim_grasp_num = test_grasp
# if (len(trajs[test_grasp - 1]) >= 3*(test_object-1) + test_trial):
#     test_data = trajs[test_grasp-1][3*(test_object-1) + test_trial - 1]
# else:
#     break

dp = len(test_data)
test_data_smoothed = np.zeros((1, sample_size))
test_scale = 1
test_x = np.linspace(0, 1, sample_size)

for i in range(0, q):
    f_smooth = interp1d(np.linspace(0, 1, dp), test_data[:, i])
    test_data_smoothed = np.append(test_data_smoothed, [f_smooth(test_x)], axis = 0)
test_data_smoothed = np.delete(test_data_smoothed, 0, axis = 0)
test_data_smoothed = test_data_smoothed.T

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

for i in range(1, sample_size):
    j = 0
    for ggr in grasp_groupings:
        for g in ggr:
            j+=1
            temp = 0
            temp2 = 0
            for k in range(0, q):
                plt.figure(num = 2)
                if (i > 1 and g == sim_grasp_num):
                    temp += weight[k] * GP_list[g-1][k].calc_likelihood(opt_scales[g-1], 0, sample_size*test_x[0:i], test_data_smoothed[0:i, k], True, 5)
                elif(i > 1):
                    temp += weight[k] * GP_list[g-1][k].calc_likelihood(opt_scales[g-1], 0, sample_size*test_x[0:i], test_data_smoothed[0:i, k], False, 5)
            llhs[g-1] = temp
            llhs_pred[g-1] = temp2
            ent_sum = 0

            if (i%1 == 0 and  i > 1):
                opt_scales[g-1] = 1
    if(i > 1):
        most_ll = 1 + np.argmax(llhs_pred + llhs)
        last_most_ll = most_ll
        most_ll_hist = np.append(most_ll_hist, most_ll)
        actual_scale_hist = np.append(actual_scale_hist, opt_scales[sim_grasp_num-1])
    f.write(','.join(['%d' %_ for _ in test_gr]))
    f.write(',')
    f.write(','.join(['%d' %_ for _ in most_ll_hist]))
    f.write('\n')
    data_formed = True
plt.ioff()
# plt.show()
# input('')
# plt.close('all')
f.close()
print('Done')
