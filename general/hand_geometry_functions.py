import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

default_phalanx_lengths = np.array([[20.0, 17.1, 12.1],
                                    [21.8, 14.1, 8.60],
                                    [24.5, 15.8, 9.80],
                                    [22.2, 15.3, 9.70],
                                    [17.2, 10.8, 8.60]])

def calculate_joint_locs(joint_angles, phalanx_lengths = default_phalanx_lengths):

    joint_locs = [[] for i in range(5)] # each item in the list contains a sequential list of joint coordinates in x, y, z

    [thumb_angles, finger_angles] =  np.split(joint_angles, [4])
    finger_angles = np.reshape(finger_angles, (4,3 ))
    [thumb_phalanx_lengths, finger_phalanx_lengths] = np.split(phalanx_lengths, [1])
    thumb_phalanx_lengths = np.squeeze(thumb_phalanx_lengths)

    # thumb joint calculations

    thumb_angles += np.array([0, np.radians(20), np.radians(10), thumb_angles[-1]]) # shift second angle by 20deg, third angle by 10deg and add the third and fourth to form a new fourth
    thumb_angles[0], thumb_angles[1] = -thumb_angles[1], -thumb_angles[0] # negate and swap the first and second joint angles

    thumb_joint_locs = np.zeros((4, 3))

    for i, (t_j_l, t_l, t_a) in enumerate(zip(thumb_joint_locs[:-1], thumb_phalanx_lengths, thumb_angles)):
        if i == 0:
            thumb_joint_locs[i+1, :2] = t_j_l[:2] + t_l * np.array([-np.cos(np.pi/6), np.sin(np.pi/6)])
        else:
            thumb_joint_locs[i+1, :2] = t_j_l[:2] + t_l * np.array([-np.cos(t_a), np.sin(t_a)])

    thumb_abd_rot_mat = np.array([[np.cos(thumb_angles[1]), 0, -np.sin(thumb_angles[1])],
                                    [0, 1, 0],
                                    [np.sin(thumb_angles[1]), 0, np.cos(thumb_angles[1])]])
    thumb_cmc_rot_mat = np.array([[np.cos(thumb_angles[0]), -np.sin(thumb_angles[0]), 0],
                                    [np.sin(thumb_angles[0]), np.cos(thumb_angles[0]), 0],
                                    [0, 0, 1]])

    thumb_joint_locs = np.matmul(thumb_abd_rot_mat, thumb_joint_locs.T).T # rotate about the thumb abduction joint
    thumb_joint_locs = np.matmul(thumb_cmc_rot_mat, thumb_joint_locs.T).T # rotate about the thumb carpometacarpal joint
    thumb_joint_locs += np.repeat(np.array([[15, -50, 0]]), 4, axis = 0)

    joint_locs[0] = thumb_joint_locs
    finger_y_shift = np.array([0, 5, 0, -5])

    for i, f_a in enumerate(finger_angles):
        finger_joint_locs = np.zeros((4, 3))
        finger_joint_locs[:, 0] = i*15 # shifting the x location of the finger over by 15 for eah finger
        finger_joint_locs[0, 1] = finger_y_shift[i] # shifting the x location of the finger over by 15 for eah finger
        for j, (f_j_l, f_l, f_a_star) in enumerate(zip(finger_joint_locs[:-1], finger_phalanx_lengths[i], np.cumsum(f_a))):
            finger_joint_locs[j+1, 1:] = f_j_l[1:] + f_l * np.array([np.cos(f_a_star), np.sin(f_a_star)])

        joint_locs[i+1] = finger_joint_locs

    return joint_locs

def plot_hand(joint_locs, _color = 'black'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for finger in joint_locs:
        ax.plot(xs = finger[:, 0], ys = finger[:, 1], zs = finger[:, 2], color = _color)

    for i in range(4):
        ax.plot(xs = [joint_locs[i][0, 0], joint_locs[i+1][0, 0]],
                ys = [joint_locs[i][0, 1], joint_locs[i+1][0, 1]],
                zs = [joint_locs[i][0, 2], joint_locs[i+1][0, 2]], color = _color)

    ax.plot(xs = [joint_locs[4][0, 0], joint_locs[4][0, 0], joint_locs[0][0, 0]],
            ys = [joint_locs[4][0, 1], joint_locs[0][0, 1], joint_locs[0][0, 1]],
            zs = [joint_locs[4][0, 2], joint_locs[0][0, 2], joint_locs[0][0, 2]], color = _color)

    axisEqual3D(ax)

    plt.show()

    return fig, ax

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
