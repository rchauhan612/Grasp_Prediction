#! /usr/bin/python

import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
import os
from hand import Finger, Thumb
from animator import get_frames

d_lengths = np.array([[21.8, 14.1, 8.60],
                    [24.5, 15.8, 9.80],
                    [22.2, 15.3, 9.70],
                    [17.2, 10.8, 8.60],
                    [20.0, 17.1, 12.1]])

subject = 'Subject 1 (Female)'
subject_path = '../HUSTdataset/Subjects/' + subject
# hand_angles = [np.array([[0, 0, 0]])] * 5
hand_angles = {}

for i in range(0, 5):
    hand_angles[i] = {}

# hand_pose = [np.zeros((4, 3))] * 5
hand_pose = {}
for i in range(0, 5):
    hand_pose[i] = {}

k = 0

for root, files, names in os.walk(subject_path, topdown = True):
    for file in files:
        # print(os.path.exists(os.path.join(subject_path, file)))
        if os.path.exists(os.path.join(subject_path, file)):
            for root1, files1, names1 in os.walk(subject_path + '/' + file, topdown = True):
                for file1 in files1:
                    for name1 in range(1, 4):
                        r = np.genfromtxt(subject_path + '/' + file + '/' + file1 + '/' + str(name1) + '.txt', delimiter='\t')
                        # t_max = len(r[:, 0])
                        # for i in range(0, t_max):
                        #     r[i, 0] = i/t_max
                        for i in range(0, 4):
                            hand_angles[i][k] = np.array([[0, 0, 0]])
                            hand_pose[i][k] = {}
                            # print(hand_angles[i])
                            f_temp = Finger(d_lengths[i, :])
                            # data_temp = np.array([[0, 0, 0]])
                            for j in range(0, len(r[:, 0])):
                                # print(hand_pose[i][k][j])
                                f_temp.set_angles(-1*r[j, (5+3*i):(8+3*i)])
                                tip_temp = np.array([[22*i, f_temp.joint_locs[3, 0], f_temp.joint_locs[3, 1]]])
                                # data_temp = np.append(data_temp, tip_temp, axis = 0)
                                hand_angles[i][k] = np.append(hand_angles[i][k], tip_temp, axis = 0)
                                hand_pose[i][k][j] = np.append(22*i*np.ones((4, 1)), f_temp.joint_locs, axis = 1)
                            # hand_pose[i] = np.append(22*i*np.ones((4, 1)), f_temp.joint_locs, axis = 1)
                        f_temp = Thumb(d_lengths[4, :])
                        hand_angles[4][k] = np.array([[0, 0, 0]])
                        hand_pose[4][k] = {}
                        for j in range(0, len(r[:, 0])):
                            # hand_angles[4][k][j] = np.zeros((4, 3))
                            f_temp.set_angles(-1*r[j, 1:5])
                            tip_temp = np.array([[-5 + f_temp.joint_locs[3, 0],
                                -30+f_temp.joint_locs[3, 1], -10+f_temp.joint_locs[3, 2]]])
                            hand_angles[4][k] = np.append(hand_angles[4][k], tip_temp, axis = 0)
                            hand_pose[4][k][j] = f_temp.joint_locs + np.repeat(np.matrix([-5, -30, -10]), 4, axis = 0)
                        # hand_pose[4] = f_temp.joint_locs + np.repeat(np.matrix([-5, -30, -10]), 4, axis = 0)
                    k += 1


                    # hand_pose[4] = np.array(hand_pose[4])
# print(hand_angles[1])

# layout = go.Layout(
#     title = subject
# )
plot_data = []
gn = 5
m = hand_angles[0][gn]
marker_size = np.linspace(1, 10, len(m))
# frames = get_frames(hand_angles[1][gn])
for i in range(0, 5):
    plot_data.append(go.Scatter3d(
        x = hand_angles[i][gn][:, 0],
        y = hand_angles[i][gn][:, 1],
        z = hand_angles[i][gn][:, 2],
        mode = 'markers',
        marker = dict(
            # color = ('rgb(0, 0, 255)'
            size = marker_size
        ),
        name = 'Finger ' + str(i+1)
    ))
    for j in range(0, len(m)-1):
        print(hand_pose[i][gn][j])

        print(i)
        plot_data.append(go.Scatter3d(
            x = hand_pose[i][gn][j][:, 0],
            y = hand_pose[i][gn][j][:, 1],
            z = hand_pose[i][gn][j][:, 2],
            mode = 'lines',
            line = dict(
                color = ('rgb(0, 0, 0)')
                ),
        ))
# f1 = frames[0]
# f1['data'][0]['type'] = 'scatter3d'
# f1['data'].append('type': 'scatter3d')
# print(f1)
# fig = {
# # [{'x': [0, 0], 'y': [0, 0], 'z': [0, 0], 'type': 'scatter3d', 'mode': 'markers'}]
#     'data': f1,
#     'layout':{'scene':{'xaxis': {'range': [-50, 50], 'autorange': False},
#               'yaxis': {'range': [-50, 50], 'autorange': False},
#               'zaxis': {'range': [-50, 50], 'autorange': False}},
#               'autorange': False,
#               'updatemenus': [{'type': 'buttons',
#                     'buttons': [{'label': 'Play',
#                         'method': 'animate',
#                         'args': [None]}]}]},
#     'frames': frames,
# }

# # print(plot_data)
# fig = go.Figure(data = plot_data, layout = layout)
# plot(fig)
plot(plot_data)

#########################################################################################

# d1 = Finger(d1_lengths)
# d1.set_angles(np.array([.7, .7, .7]))
#
# finger_data.extend([go.Scatter(
#     x = d1.joint_locs[:, 0],
#     y = d1.joint_locs[:, 1],
#     mode = 'lines',
#     name = 'finger'
# )])
