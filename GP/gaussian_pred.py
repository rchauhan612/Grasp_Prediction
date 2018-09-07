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

for i in range(0, 33):
    for j in range(0, len(trajs[i])):
        trajs[i][j] = np.matmul(trajs[i][j][:, 1:d+1], PC)
        new_traj = np.zeros((sample_size, 1));
        for row in trajs[i][j].T:
            new_traj = np.append(new_traj, signal.resample(row, sample_size).reshape((1, sample_size)).T, axis = 1)
        new_traj = np.delete(new_traj, 0, 1)
        trajs[i][j] = new_traj
    temp_var = np.zeros((q, sample_size))
    temp_mean = np.zeros((q, sample_size))
    if i == 20:
        plt.figure(num = i)
    handles = [None for _ in range(q)]
    for w in range(0, q):
        if i == 20:
            for k in range(0, len(trajs[i])):
                if k == 0:
                    handles[w] = plt.scatter(np.linspace(0, 1, sample_size), trajs[i][k][:, w], s = 7, color = colors[w], alpha = 1)
                else:
                    plt.scatter(np.linspace(0, 1, sample_size), trajs[i][k][:, w], s = 7, color = colors[w])
        temp = np.zeros((len(trajs[i]), sample_size))
        for j in range(0, sample_size):
            for k in range(0, len(trajs[i])):
                temp[k, j] = trajs[i][k][j, w]
        temp_var[w, :] = np.var(temp, axis = 0)
        temp_mean[w, :] = np.mean(temp, axis = 0)
    temp = [temp_var, temp_mean]
    # plt.title('Grasp %d Trajectory' % (i+1))
    if i == 20:
        plt.xlabel('Grasp Progress (%)')
        plt.xlim([0, 1])
        plt.ylabel('PC Magnitude')
        plt.legend(handles, ['PC%d' % (_) for _ in range(1, q+1)], ncol = q, loc = 9, facecolor = 'white', bbox_to_anchor = (.5, 1.15), handlelength = .5)
        plt.show()
    grasp_set.append(temp)
# input('')
# print(len(trajs), len(trajs[0]))

x_star_plt = np.linspace(0.1, sample_size, 1000)
ggr_num = 1

prog = 0
layout = (7, 5)
fig = plt.figure(num=1, figsize=(layout[1]*220/60, layout[0]*195/60), dpi=57)
plt.ion()
plt.show()
pltcnt = 0
GP_list = [[None for _ in range(q)] for _ in range(33)]
gp_x_plot_data = np.linspace(0, 1, len(x_star_plt))
gp_y_plot_data = [[[None for _ in range(3)] for _ in range(q)] for _ in range(33)]
for w in range(0, len(grasp_groupings)):
    ggr = grasp_groupings[w]
    ggr_num+=1
    for g in range(0, len(ggr)):
        pltcnt+=1
        for i in range(0, q):
            prog+=1
            gp = GP(np.arange(0, sample_size), np.interp(np.arange(0, sample_size), np.linspace(0, sample_size, len(grasp_set[ggr[g]-1][0][i, :])), grasp_set[ggr[g]-1][0][i, :])) # mean(grasp_set[ggr[g]-1][0][i, :]))
            gp.opt(grasp_set[ggr[g]-1][1][i, :])
            x_star = np.linspace(0.1, sample_size, 50)
            (y_bar_star, var_y) = gp.draw_sample(x_star)
            gp.eval_continuous(50)
            print("%.2f%%" % (100*prog/(33*q)))
            GP_list[ggr[g]-1][i] = gp
            var_y_smooth = interp1d(x_star, var_y, kind='cubic')
            var_y = var_y_smooth(x_star_plt)
            y_bar_star_smooth = interp1d(x_star, y_bar_star, kind='cubic')
            y_bar_star = y_bar_star_smooth(x_star_plt)
            var_y_top = y_bar_star+1*sqrt(var_y)
            var_y_bot = y_bar_star-1*sqrt(var_y)
            gp_y_plot_data[ggr[g]-1][i][0] = y_bar_star
            gp_y_plot_data[ggr[g]-1][i][1] = var_y_top
            gp_y_plot_data[ggr[g]-1][i][2] = var_y_bot
            if (ggr[g] == 21):
                plt.figure(num = 111)
                plt.fill_between(gp_x_plot_data, var_y_top, var_y_bot, color=colors[i], alpha=0.2)
                plt.plot(gp_x_plot_data, y_bar_star, color=colors[i], linestyle = 'dashed')
                plt.xlabel('Grasp Progress (%)')
                plt.xlim([0, 1])
                plt.ylabel('PC Magnitude')
                # plt.legend(handles, ['PC%d' % (_) for _ in range(1, q+1)], ncol = q, loc = 9, facecolor = 'white', bbox_to_anchor = (.5, 1.15), handlelength = .5)
                plt.show()
            plt.figure(num = 1)
            plt.subplot(layout[0], layout[1], pltcnt)
            plt.fill_between(gp_x_plot_data, var_y_top, var_y_bot, color=colors[i], alpha=0.2)
            plt.plot(gp_x_plot_data, y_bar_star, color=colors[i], linestyle = 'dashed')
            plt.ylim([-3, 3.5])
            plt.xlim([0, 1])
            plt.yticks(np.arange(-3, 4, step=0.5))
            if (g!=0 and g!=layout[1]):
                plt.gca().set_yticklabels([])
        plt.title(grasp_names[ggr[g]-1])
        plt.tight_layout()
fig.canvas.draw()
plt.pause(0.001)
plt.show()
plt.figure(num = 2, figsize=(10, 5), dpi = 60)
# input('')
# sim_grasp = np.random.choice(grasp_names)
# sim_grasp_num = int(sim_grasp.split(" ")[0])
sim_grasp_num = test_grasp
# for root, files, names in os.walk(subject_path + '/' + sim_grasp, topdown = True):
#     sim_obj = np.random.choice(files)
#     for root1, files1, names1 in os.walk(subject_path + '/' + sim_grasp + '/' + sim_obj, topdown = True):
#         names1 = [name for name in names1 if name.endswith('.txt')]
#         sim_trial = np.random.choice(names1)
#         break
#     break

# test_path = subject_path + '/' + sim_grasp + '/' + sim_obj + '/' + sim_trial
# test_data = np.genfromtxt(test_path, delimiter='\t')
# test_data = np.delete(test_data, [0, 17], axis = 1)
# test_data = np.matmul(test_data , PC) #- mean(test_data, axis = 0)
# print(test_grasp-1, 3*(test_object-1) + test_trial)
if (len(trajs[test_grasp - 1]) >= 3*(test_object-1) + test_trial):
    test_data = trajs[test_grasp-1][3*(test_object-1) + test_trial - 1]
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
#

# plt.subplot(1, 2, 1)
# for k in range(q):
#     plt.plot(gp_x_plot_data, gp_y_plot_data[sim_grasp_num-1][k][0], color = colors[k], linestyle = 'dashed')
#     plt.fill_between(gp_x_plot_data, gp_y_plot_data[sim_grasp_num-1][k][1], gp_y_plot_data[sim_grasp_num-1][k][2],
#         color=colors[k], alpha=0.2)
# plt.title(sim_grasp)
# plt.ylim([-3, 3.5])
# plt.xlim([0, 1])
# plt.yticks(np.arange(-3, 4, step=0.5))
#
# plt.subplot(1, 2, 2)
# for k in range(q):
#     most_ll_gp_set[0][k] = plt.plot(gp_x_plot_data, gp_y_plot_data[sim_grasp_num-1][k][0], color = colors[k], linestyle = 'dashed')
#     most_ll_gp_set[1][k] = plt.fill_between(gp_x_plot_data, gp_y_plot_data[sim_grasp_num-1][k][1], gp_y_plot_data[sim_grasp_num-1][k][2],
#         color=colors[k], alpha=0.2)
# plt.title(sim_grasp_num)
# plt.ylim([-3, 3.5])
# plt.xlim([0, 1])
# plt.yticks(np.arange(-3, 4, step=0.5))

opt_PC_cnt = 1


last_most_ll = sim_grasp_num
most_ll = sim_grasp_num
most_ll_hist = np.array([])
actual_scale_hist = np.array([])

for i in range(1, sample_size):
    j = 0
    # print(most_ll)
    # if (i > 1):
    #     for m in range(q):
            # if (i%1 == 0):
            # test_GP[m].add_prior(test_x[i]/test_scale*sample_size, test_data_smoothed[i, m])
            # test_GP[m].add_prior(test_x[i]*opt_scales[most_ll-1], test_data_smoothed[i, m])
            # test_pred_x = np.arange(test_x[i]/opt_scales[most_ll-i], 1, 0.05)*sample_size
            # test_pred_pt[m] = test_GP[m].draw_sample(test_pred_x)
            # test_pred_x = test_pred_x / sample_size
            # print(test_pred_x)
            # test_pred_pt[m] = test_GP[m].draw_sample(test_x[i:]/opt_scales[most_ll-i]*sample_size)
            # test_pred_pt[m] = test_GP[m].draw_sample(test_x[i:]*opt_scales[most_ll-1])
    for ggr in grasp_groupings:
        for g in ggr:
            j+=1
            # ax = plt.subplot(layout[0], layout[1], j)
            # if (i > 1):
            #     llhs_text[g-1].set_text('L=%.3f' % llhs[g-1])
            #     llhs_pred_text[g-1].set_text('L Pred=%.3f' % llhs_pred[g-1])
            #     # ents_text[g-1].set_text('KL=%.3f' % ents[g-1])
            #     if (g == (1+np.argmax(llhs+llhs_pred))):
            #         plt.setp(ax.spines.values(), linewidth=3)
            #         ax.spines['top'].set_color('green')
            #         ax.spines['bottom'].set_color('green')
            #         ax.spines['left'].set_color('green')
            #         ax.spines['right'].set_color('green')
            #     else:
            #         plt.setp(ax.spines.values(), linewidth=1)
            #         ax.spines['top'].set_color('black')
            #         ax.spines['bottom'].set_color('black')
            #         ax.spines['left'].set_color('black')
            #         ax.spines['right'].set_color('black')
            # else:
            #     llhs_text[g-1] = plt.text(0.1, -2, ' ', weight = 'bold')
            #     llhs_pred_text[g-1] = plt.text(0.1, -2.5, ' ', weight = 'bold')
            #     # ents_text[g-1] = plt.text(0.1, -2.5, ' ', weight = 'bold')
            # # print(test_data_smoothed[:, i])
            # if (g == sim_grasp_num):
            #     plt.setp(ax.spines.values(), linewidth=3)
            #     ax.spines['top'].set_color('red')
            #     ax.spines['bottom'].set_color('red')
            #     ax.spines['left'].set_color('red')
            #     ax.spines['right'].set_color('red')
            temp = 0
            temp2 = 0
            for k in range(0, q):
                plt.figure(num = 2)
                # plt.scatter(test_x[i], test_data_smoothed[i, k], color = colors[k], s = 7)
                # if (i > 2):
                #     proj_set[g-1][k].remove()
                if (i > 1 and g == sim_grasp_num):
                    # proj_set[g-1][k] = plt.scatter(test_x[i+1:-1], test_pred_pt[k][0], color = 'black', s = 7)
                    temp += weight[k] * GP_list[g-1][k].calc_likelihood(opt_scales[g-1], 0, sample_size*test_x[0:i], test_data_smoothed[0:i, k], True, 5)
                    # temp2 += weight[k] * GP_list[g-1][k].calc_likelihood(opt_scales[g-1], 0, sample_size*test_pred_x, test_pred_pt[k][0], True, 3)
                elif(i > 1):
                    # proj_set[g-1][k] = plt.scatter(test_x[i+1:-1], test_pred_pt[k][0], color = 'black', s = 7)
                    temp += weight[k] * GP_list[g-1][k].calc_likelihood(opt_scales[g-1], 0, sample_size*test_x[0:i], test_data_smoothed[0:i, k], False, 5)
                    # temp2 += weight[k] * GP_list[g-1][k].calc_likelihood(opt_scales[g-1], 0, sample_size*test_pred_x, test_pred_pt[k][0], False, 3)
            llhs[g-1] = temp
            llhs_pred[g-1] = temp2
            ent_sum = 0

            if (i%1 == 0 and  i > 1):
                opt_scales[g-1] = 1
                # opt_scales[g-1] = opt_gr_scale(1, weight[0:opt_PC_cnt], GP_list[g-1], sample_size*test_x[:i], [test_data_smoothed[:i, k] for k in range(opt_PC_cnt)])
                # opt_scales[g-1] = opt_gr_scale(1, weight[0:opt_PC_cnt], GP_list[g-1], np.append(sample_size*test_x[:i], test_pred_x), [np.append(test_data_smoothed[:i, k], test_pred_pt[k][0]) for k in range(opt_PC_cnt)])
    if(i > 1):
        # plt.subplot(1, 2, 2)
        # plt.cla()
        most_ll = 1 + np.argmax(llhs_pred + llhs)
        # if (most_ll != last_most_ll):
        last_most_ll = most_ll

        # for k in range(q):
        #     plt.subplot(1, 2, 1)
        #     if (i > 2):
        #         prior_set[0][k].remove()
        #         # proj_set[0][k].remove()
        #     prior_set[0][k] = plt.scatter(test_x[0:i]/opt_scales[sim_grasp_num-1], test_data_smoothed[0:i, k], color = colors[k], s = 7)
        #     # proj_set[0][k] = plt.scatter(test_pred_x*opt_scales[most_ll -1] / opt_scales[sim_grasp_num-1], test_pred_pt[k][0], color = 'black', s = 7)
        #     plt.xlim([0, max(test_x[i:]/opt_scales[sim_grasp_num-1])])
        #
        #     plt.subplot(1, 2, 2)
        #     # if (i > 2):
        #         # prior_set[1][k].remove()
        #     prior_set[1][k] = plt.scatter(test_x[0:i]/opt_scales[most_ll-1], test_data_smoothed[0:i, k], color = colors[k], s = 7)
        #     # proj_set[1][k] = plt.scatter(test_pred_x, test_pred_pt[k][0], color = 'black', s = 7)
        #
        # plt.subplot(1, 2, 2)
        # plt.cla()
        # for k in range(q):
        #     # print(most_ll_gp_set[0][k])
        #     # most_ll_gp_set[0][k].remove()
        #     most_ll_gp_set[0][k] = plt.plot(gp_x_plot_data, gp_y_plot_data[most_ll-1][k][0], color = colors[k], linestyle = 'dashed')
        #     # most_ll_gp_set[1][k].remove()
        #     most_ll_gp_set[1][k] = plt.fill_between(gp_x_plot_data, gp_y_plot_data[most_ll-1][k][1], gp_y_plot_data[most_ll-1][k][2],
        #         color=colors[k], alpha=0.2)
        # plt.title(str(most_ll))
        # plt.ylim([-3, 3.5])
        # plt.xlim([0, max(test_x[i:]/opt_scales[most_ll-1])])
        # plt.yticks(np.arange(-3, 4, step=0.5))

        # print('-----------------------%d%%-----------------------' % (100*i/(sample_size-1)))
        # print('Prior:\tMost Likely: %d, %.3f\tActual: %.3f' % (1+np.argmax(llhs), max(llhs), llhs[sim_grasp_num-1]))
        # print('Pred:\tMost Likely: %d, %.3f\tActual: %.3f' % (1+np.argmax(llhs_pred), max(llhs_pred), llhs_pred[sim_grasp_num-1]))
        # print('Most Likely Scale: %.3f\tActual Grasp Scale: %.3f, %.3f' % (opt_scales[np.argmax(llhs_pred+llhs)], opt_scales[sim_grasp_num-1], test_scale))
        # print('-----')
        # print('Total:\tMost Likely: %d, %.3f\tActual: %d, %.3f' % (most_ll, max(llhs_pred+llhs), sim_grasp_num, llhs[sim_grasp_num-1] + llhs_pred[sim_grasp_num-1]))
        # for val in opt_scales:
        #     print(val)
        most_ll_hist = np.append(most_ll_hist, most_ll)
        actual_scale_hist = np.append(actual_scale_hist, opt_scales[sim_grasp_num-1])
        # print(1+np.argmin(ents))
    # fig.canvas.draw_idle()
    # plt.pause(.00001)
    # plt.show()

    # plt.ioff()
    #
    # plt.figure(num = 3)
    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(len(most_ll_hist)), most_ll_hist)
    # plt.title('most_ll grasp')
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(len(most_ll_hist)), actual_scale_hist)
    # plt.title('scale of actual')

    # plt.show()
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
