#! /usr/bin/python

import numpy as np
import collections
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.axes_grid1 import make_axes_locatable

grasp_groupings = np.array([[1, 2, 3], [11, 13, 14, 26, 27, 28], [9, 24, 31, 33],
    [6, 7, 8, 20, 25, 21, 23], [17, 19, 29, 32], [10, 12], [18, 22], [4, 5, 15, 16, 30]])

file_name = 'results_actual measurement variance_6PC.csv'

orig_data = np.genfromtxt(file_name, delimiter = ',')
subjects = orig_data[:, 0]
correct_grasps = orig_data[:, 1]
ml = orig_data[:, 4:]

u, indices = np.unique(subjects, return_index=True)

data = []

for i in range(1, len(indices)):
    c1 = indices[i-1]
    c2 = indices[i]
    data.append((correct_grasps[c1:c2], ml[c1:c2, :]))

# fig, (p1, p2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios': [3, 1]}, figsize = (.8*6, .8*9))
# plt.subplots_adjust(left = None, bottom = 0.1, right = None, top = 0.95)
# plt.ion()
# p1.set_axisbelow(True)
# fig.canvas.draw()
# time_sections = np.arange(1, 19)
time_sections = [6]

errors2_hist2 = np.linspace(0, 100, 18)[None].T
errors_hist2 = np.linspace(0, 100, 18)[None].T
# print(errors2_hist2)
for vis_subject in range(1, 2):
    # p2.cla()
    # p2.set_xlim(0, 100)
    # p2.set_ylim(0, 300)
    # p2.set_xlabel('Grasp Completion')
    # p2.set_ylabel('Errors')
    # p1.cla()
    # p1.set_aspect('equal')
    # p1.grid(b = True, which = 'major', linestyle = '-', color = '#C2C2C2')
    # p1.minorticks_on()
    # p1.grid(b = True, which = 'minor', linestyle = '--', color = '#C2C2C2')
    # p1.set_xlabel('Test Grasp Number')
    # p1.set_ylabel('Predicted Grasp Number')
    # p1.set_xlim(0, 34)
    # p1.set_ylim(0, 34)
    # print(vis_subject)
    errors2_hist = []
    errors_hist = []
    for ts in range(len(time_sections)):
        plt_data = data[vis_subject][1][:, time_sections[ts]-1]
        temp = []
        for i in range(1, 34):
            temp.append(plt_data[data[vis_subject][0] == i])

        plt_data = temp
        occurs = []
        errors = 0
        errors2 = 0
        max_occurs = 9
        for i in range(len(plt_data)):
            temp = []
            errors3 = 0
            errors4 = 0
            for j in range(len(plt_data[i])):
                temp.append((plt_data[i] == plt_data[i][j]).sum())
                if(plt_data[i][j] != i+1):
                    errors += 1
                    errors4 += 1
                for group in grasp_groupings:
                    group = np.array(group)
                    if (group == (i+1)).sum() > 0:
                        if (group == plt_data[i][j]).sum() == 0:
                            errors2 += 1
                            errors3 += 1
            print(i+1, errors4)
            occurs.append(temp)

        # p1.cla()
        # for i in range(len(plt_data)):
        #     for j in range(len(plt_data[i])):
        #         p1.scatter(i+1, plt_data[i][j], s = 30, facecolors = 'none', edgecolors = cm.winter(occurs[i][j] / max_occurs))

        # p1.set_title('Subject: %d Time = %.2f%%' % (vis_subject, 100*time_sections[ts]/(time_sections[-1]-1)))

        errors2_hist.append(errors2)
        errors_hist.append(errors)

        # print(errors, errors2)
        # fig.canvas.draw()
        # plt.show()
        # plt.pause(0.01)

        # p2.scatter(100*ts/len(time_sections), errors, s = 20, facecolors = 'none', edgecolors = 'orange')
        # p2.scatter(100*ts/len(time_sections), errors2, s = 20, facecolors = 'none', edgecolors = 'blue')

    errors_hist2 = np.concatenate((errors_hist2, np.array(errors_hist)[None].T), axis = 1)
    errors2_hist2 = np.concatenate((errors2_hist2, np.array(errors2_hist)[None].T), axis = 1)
    # print(errors_hist2)
    # time.sleep(1)
# plt.ioff()
# print(100-100*np.mean(errors_hist) / 297, 100-100*np.mean(errors2_hist)/297)
# print(np.std(100*np.array(errors_hist)/297), np.std(100*np.array(errors2_hist)/297))
# plt.show()
e_mean = np.zeros(18)
e2_mean = np.zeros(18)
e_std = np.zeros(18)
e2_std = np.zeros(18)
for i in range(0, 18):
    e_mean[i] = np.mean(errors_hist2[i, 1:])
    e2_mean[i] = np.mean(errors2_hist2[i, 1:])
    e_std[i] = np.std(errors_hist2[i, 1:])
    e2_std[i] = np.std(errors2_hist2[i, 1:])
res = np.concatenate((np.array(e_mean)[None], np.array(e_std)[None], np.array(e2_mean)[None], np.array(e2_std)[None]), axis = 0)
print(res.shape)
np.savetxt('overtime.csv', res, delimiter = ',')
# print(e_mean, e_std, e2_mean, e2_std)
