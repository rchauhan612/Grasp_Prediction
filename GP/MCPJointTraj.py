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

subject = 'Subject 4 (Female)'
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
gn = 32
gname = []

plot_data = []
finger_markers = ['circle', 'square', 'diamond', 'triangle', 'hexagon']
obj_colors = ['rgb(65, 162, 252)', 'rgb(244, 153, 24)', 'rgb(7, 175, 38)', 'rgb(175, 15, 7)', 'rgb(39, 80, 163)']

for root, files, names in os.walk(subject_path, topdown = True):
    for file in files:
        # print(file)
        # print(os.path.exists(os.path.join(subject_path, file)))
        if (os.path.exists(os.path.join(subject_path, file))):
            gname.append(file)
            print(file)
            for root1, files1, names1 in os.walk(subject_path + '/' + file, topdown = True):
                grasp_data = []
                # print(files1)
                cnt = 0
                for file1 in files1:
                    temp_data = []
                    # print(file1)
                    for name1 in range(1, 4):
                        # print(file1 + '/' + str(name1) + '.txt')
                        r = np.genfromtxt(subject_path + '/' + file + '/' + file1 + '/' + str(name1) + '.txt', delimiter='\t')
                        for i in range(0, 4):
                            hand_angles[i][k] = np.array([[0, 0, 0, 0]])
                            for j in range(0, len(r[:, 0])):
                                temp = np.append(r[j, 0], 180*r[j, (5+3*i):(8+3*i)] / np.pi)
                                hand_angles[i][k] = np.append(hand_angles[i][k], [temp], axis = 0)
                        hand_angles[4][k] = np.array([[0, 0, 0, 0]])
                        for j in range(0, len(r[:, 0])):
                            temp = np.append(r[j, 0], 180*r[j, 1:4] / np.pi)
                            hand_angles[4][k] = np.append(hand_angles[4][k], [temp], axis = 0)
                        k += 1
                    for i in range(0, 5):
                        x_temp = []
                        y_temp = []
                        for j in range(-3, 0):
                            x_temp=(hand_angles[i][k+j][:, 0].flatten())
                            y_temp=(hand_angles[i][k+j][:, 1].flatten())
                            grasp_data.append(go.Scatter(
                                x = x_temp,
                                y = y_temp,
                                mode = 'markers',
                                name = file1 + ', Digit: ' + str(i),
                                showlegend = (j==-3),
                                legendgroup = 'Digit ' + str(i),
                                marker = dict(
                                    symbol = finger_markers[cnt],
                                    color = (obj_colors[i]),
                                    # size = marker_size

                                ),
                            ))
                    cnt += 1
                # temp_data.extend(grasp_data)
                plot_data.extend(grasp_data)
        # print(len(temp_data))

dropdown_data = [];
for i in range(0, 33):
    dropdown_data.append(
            dict(label = gname[i],
                 method = 'update',
                 args = [{'visible': np.repeat(np.arange(33), 45) == i},
                         # {'title': gname[i]}
                         ])
    )
print(len(plot_data))
updatemenus = list([
    dict(active=-1,
         buttons=dropdown_data)])
layout = go.Layout(
    title = subject,
    showlegend = True,
    updatemenus = updatemenus,
    xaxis = dict(
        title = 'Time (s)'
    ),
    yaxis = dict(
        title = 'Angle (deg)'
    )
)

fig = go.Figure(data = plot_data, layout = layout)
plot(fig, filename = subject + '.html')
