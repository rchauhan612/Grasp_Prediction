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

class GP:
    def __init__(self, x, var):
        self.x = x
        self.n = len(x)
        self.var = var
        self.hypr_L = 2
        self.tscale = 1
        self.K = np.ones((2, 2))
        self.orig_scale = x[-1]
        # print(len(x), len(var))

    def kron_del(self, x0, x1):
        return x0==x1

    def kern_fun(self, x0, x1, hypr):
        temp = self.var[np.argmin(abs(self.x-x0))]
        return (self.hypr_L**2) * exp((-np.linalg.norm(x0-x1)**2) / (2*hypr[0]**2)) + (temp)*self.kron_del(x0, x1)

    def get_kern(self, hypr):
        K = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            for j in range(0, self.n):
                K[i, j] = self.kern_fun(self.x[i], self.x[j], hypr)
        # print(K)
        return K

    def log_pdf(self, hypr):
        lpdf = -.5 * (np.matmul(np.matmul(self.y.T , np.linalg.inv(self.get_kern(hypr))) , self.y) - log(np.linalg.norm(self.get_kern(hypr))) - self.n * log(2*pi)) #this has been negated so that minimize actually finds the maximum
        # print(lpdf)
        return -1 * lpdf

    def add_prior(self, x, y):
        self.x = np.append(self.x, x)
        self.n = len(self.x)
        self.opt(np.append(self.y, y))
        # print(self.x, self.y)

    def opt(self, y):
        self.y = y
        self.hypr = [10]
        # hypr0 = [1]
        # sol = minimize(self.log_pdf, hypr0, method='CG', options={'maxiter': 20})
        # # val = sol.fun
        # self.hypr = sol.x
        # self.hypr = [10, 10]
        old_K = self.K
        self.K = self.get_kern(self.hypr)
        # if np.isfinite(np.linalg.cond(self.K)):
        self.K_inv = np.linalg.inv(self.K)
            # self.K = old_K
        self.sample_values_x = np.linspace(self.x[0], self.x[-1], 100)
        self.sample_values_y = self.draw_sample(self.sample_values_x)
        # print(len(self.sample_values_y[1]))

    def get_K_star(self, x_star):
        temp = []
        for j in self.x:
            temp.append(self.kern_fun(x_star, j, self.hypr))
        self.K_star = np.array(temp)
        return np.array(temp)

        # self.K_star = np.array(self.K_star[1:len(x_star)+1])

    def get_K_starstar(self, x_star):
        self.K_starstar = self.kern_fun(x_star, x_star, self.hypr)
        return self.kern_fun(x_star, x_star, self.hypr)

    def get_hypr(self):
        return (self.hypr_L, self.hypr[0], sqrt(mean(self.var)))

    def eval_continuous(self, density):
        self.cont_sample = self.draw_sample(np.linspace(0, self.orig_scale, density))

    def draw_sample(self, x_star):
        y_bar_star = [None for _ in range(len(x_star))]
        var_y_star = [None for _ in range(len(x_star))]
        for i in range(len(x_star)):
            K_star = self.get_K_star(x_star[i])
            K_starstar = self.get_K_starstar(x_star[i])
            y_bar_star[i] = np.matmul(np.matmul(self.K_star, self.K_inv), self.y)
            var_y_star[i] = K_starstar - np.matmul(np.matmul(K_star, self.K_inv), K_star.T)
        return (y_bar_star, abs(np.array(var_y_star)))

    def calc_likelihood(self, scale, shift, x_eval, y_eval, show, mode):
        ll = 0
        # print(scale)
        if mode==1:
            x_eval_scale = (x_eval*scale) + shift
            (y_bar_star, var_y_star) = self.draw_sample(x_eval_scale)
            for i in range(len(x_eval)):
                s = (y_eval[i] - y_bar_star[i])**2
                ll_temp = min(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), 1)
                ll += 2 * ll_temp * exp(-(i-len(x_eval))**2 / (2*len(x_eval))) / sqrt(2*pi*len(x_eval))
        elif mode==2:
            x_eval_scale = (x_eval*scale) + shift
            (y_bar_star, var_y_star) = self.draw_sample(x_eval_scale)
            for i in range(len(x_eval)):
                s = (y_eval[i] - y_bar_star[i])**2
                ll_temp = min(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), 1)
                ll += 2 * ll_temp * exp(-(i)**2 / (2*len(x_eval))) / sqrt(2*pi*len(x_eval))
        elif mode==3:
            gp_scale = self.sample_values_x/scale + shift
            # print(scale, gp_scale[0], gp_scale[-1])
            for i in range(len(x_eval)):
                nearest = self.get_closest(gp_scale, self.sample_values_y[0], x_eval[i])
                if nearest[0]:
                    y_bar_star = self.sample_values_y[0][nearest[1]]
                    var_y_star = self.sample_values_y[1][nearest[1]]
                    s = (y_eval[i] - y_bar_star)**2
                    # print(x_eval[i], nearest[2])
                    # print(y_eval[i], y_bar_star)
                    ll_temp = min(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), 1)
                else:
                    ll_temp = 0
                ll += 2 * ll_temp * exp(-(i)**2 / (2*len(x_eval))) / sqrt(2*pi*len(x_eval))
        elif mode==4:
            gp_scale = self.sample_values_x/scale + shift
            # print(scale, gp_scale[0], gp_scale[-1])
            for i in range(len(x_eval)):
                nearest = self.get_closest(gp_scale, self.sample_values_y[0], x_eval[i])
                if nearest[0]:
                    y_bar_star = self.sample_values_y[0][nearest[1]]
                    var_y_star = self.sample_values_y[1][nearest[1]]
                    s = (y_eval[i] - y_bar_star)**2
                    # print(x_eval[i], nearest[2])
                    # print(y_eval[i], y_bar_star)
                    ll_temp = min(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), 1)
                else:
                    ll_temp = 0
                ll += ll_temp
        elif mode == 5:
            gp_scale = self.sample_values_x/scale + shift
            nearest = self.get_closest(gp_scale, self.sample_values_y[0], x_eval)
            var_y_star = self.sample_values_y[1][nearest[1]]
            y_bar_star = nearest[3]
            s = (y_eval - y_bar_star)**2
            ll = sum(np.clip(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), None, 1))
        # print(ll)
        return abs(ll)

    def get_closest(self, arr_x, arr_y, target):
        if not isinstance(target, (list, tuple, np.ndarray)):
            loc = np.argmin(abs(arr_x - target))
            return(True, np.array([loc]), np.array([arr_x[loc]]), np.array([arr_y[loc]]))
        else:
            temp_loc = []
            temp_x = []
            temp_y = []
            for i in target:
                temp_loc.append(np.argmin(abs(arr_x - i)))
                temp_x.append(arr_x[temp_loc[-1]])
                temp_y.append(arr_y[temp_loc[-1]])
            return(True, np.array(temp_loc), np.array(temp_x), np.array(temp_y))

    def plot_pt(self, t, pt, m, S):
        x = m + np.linspace(-3*sqrt(S), 3*sqrt(S), 50)
        f = exp(-((x-m)**2) / (2*S)) / sqrt(2*pi*S)
        plt.figure(num = 2)
        # plt.subplot(layout[0], layout[1], 35)
        plt.ion()
        input()
        plt.clf()
        plt.plot(x, f)
        plt.title('t: %.3f' % t)
        plt.axvline(x = pt, color = 'red')
        fig.canvas.draw()
        plt.pause(0.001)
        plt.show()
        fig.canvas.draw()

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
# f.close()
for subject in np.arange(1, 31):
    data_formed = False
    for test_grasp in np.arange(1, 2):
        for test_object in np.arange(1, 2):
            for test_trial in np.arange(1, 2):

                    # for subject in np.arange(1, 2):
                    #     for test_grasp in np.arange(1, 2):
                    #         for test_object in np.arange(1, 2):
                    #             for test_trial in np.arange(1, 2):
                k = 0
                d = 16
                data = np.zeros((1, d))
                data_f = data
                q = 6

                grasp_set = []
                grasp_names = []
                grasp_groupings = [[1, 2, 3], [11, 13, 14, 26, 27, 28], [9, 24, 31, 33],
                    [6, 7, 8, 20, 25, 21, 23], [17, 19, 29, 32], [10, 12], [18, 22], [4, 5, 15, 16, 30]]

                # test_gr = [np.random.choice(np.arange(1, 30)),
                #     np.random.choice(np.arange(1, 34)),
                #     np.random.choice(np.arange(1, 4)),
                #     np.random.choice(np.arange(1, 4))]

                test_gr = [subject, test_grasp, test_object, test_trial]
# test_gr = [1, 1, 1, 2]

                print(test_gr)
                if(not data_formed):

                    trajs = []
                    for root0, files0, names0 in os.walk('../HUSTdataset/Subjects', topdown = True):
                        cnt0 = 0
                        for file0 in files0:
                            if (os.path.exists(os.path.join('../HUSTdataset/Subjects', file0))):
                                # print(file0)
                                cnt0 += 1
                                if (int(file0.split(' ')[1]) == test_gr[0]):
                                    subject_path = '../HUSTdataset/Subjects' + '/' + file0
                                    for root, files, names in os.walk(subject_path, topdown = True):
                                        cnt = 0
                                        for file in files:
                                            if (os.path.exists(os.path.join(subject_path, file))):
                                                cnt+=1
                                                grasp_names.append(file)
                                                if (cnt == test_gr[1]):
                                                    sim_grasp = file
                                                for root1, files1, names1 in os.walk(subject_path + '/' + file, topdown = True):
                                                    cnt2 = 0
                                                    grasp_trajs = []
                                                    for file1 in files1:
                                                        cnt2+=1
                                                        for name1 in range(1, 4):
                                                            r = np.genfromtxt(subject_path + '/' + file + '/' + file1 + '/' + str(name1) + '.txt', delimiter='\t')
                                                            # if (cnt == test_gr[1] and cnt2 == test_gr[2] and name1 == test_gr[3]):
                                                                # test_data = r
                                                                # print(subject_path + '/' + file + '/' + file1 + '/' + str(name1))
                                                            grasp_trajs.append(r)
                                                            data = np.append(data, r[:, 1:d+1], axis=0) # all trajectory points
                                                            data_f = np.append(data_f, [r[r.shape[0]-1, 1:d+1]], axis=0) # only end configuration
                                                    if (len(grasp_trajs) > 0):
                                                        trajs.append(grasp_trajs)

                    data = data_f
                    data = np.delete(data, 0, axis = 0)
                    data_f = np.delete(data_f, 0, axis = 0)
                    n = data.shape[0]
                    avg = np.mean(data, axis = 0)
                    X = data - np.repeat([avg], n, axis = 0)
                    C = np.matmul(X.T, X)
                    weight, PC = np.linalg.eig(C)
                    weight = weight / sum(weight)
                    # weight = (1/q) * np.ones(q)
                    np.save('PCAs/'+str(subject), PC)
                    PC = PC[:, 0:q]

                    sample_size = 20
                    colors = pl.cm.tab10(np.linspace(0,0.6,q))

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
                else:
                    break

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
