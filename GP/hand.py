import numpy as np

class Finger:
    def __init__(self, link_lengths):
        self.lengths = link_lengths
        self.angles = np.array([0, 0, 0])

    def get_joint_locs(self):
        joint_locs = np.zeros((4, 2))
        for i  in range(0, 3):
            l = self.lengths[i]
            th = 0
            for j in range(0, i+1):
                th += self.angles[j]
            jx = joint_locs[i, 0] + l * np.cos(th)
            jy = joint_locs[i, 1] + l * np.sin(th)
            joint_locs[i+1, :] = np.array([jx, jy])
        self.joint_locs = joint_locs

    def set_angles(self, angles):
        self.angles = angles
        self.get_joint_locs()

class Thumb:
    def __init__(self, link_lengths):
        self.lengths = link_lengths
        self.angles = np.array([0, 0, 0])

    def get_joint_locs(self):
        joint_locs = np.zeros((4, 3))

        angles_temp = self.angles + np.array([0, 0.3491, 0, 0])
        temp = angles_temp[0];
        angles_temp[0] = angles_temp[1];
        angles_temp[1] = temp;

        angles_temp[0] *= -1;
        angles_temp[1] *= -1;
        angles_temp[2] +=  0.1745;
        angles_temp[3] += angles_temp[2];

        joint_locs[1, :] = joint_locs[0, :] + self.lengths[0] * np.array([0, -np.cos(np.pi/6), np.sin(np.pi/6)]);

        for i  in range(1, 3):
            l = self.lengths[i]
            jx = joint_locs[i, 1] + l * -np.cos(angles_temp[i])
            jy = joint_locs[i, 2] + l * np.sin(angles_temp[i])
            joint_locs[i+1, :] = np.array([0, jx, jy])

        joint_locs = np.transpose(joint_locs)
        angles_temp[1] += np.pi/2
        angles_temp[0] -= np.pi
        joint_locs = np.matrix([[np.cos(angles_temp[1]), 0, -np.sin(angles_temp[1])],
                                [0, 1, 0],
                                [np.sin(angles_temp[1]), 0, np.cos(angles_temp[1])]]) * joint_locs;
        joint_locs = np.matrix([[np.cos(angles_temp[0]), -np.sin(angles_temp[0]), 0],
                                [np.sin(angles_temp[0]), np.cos(angles_temp[0]), 0],
                                [0, 0, 1]]) * joint_locs;

        joint_locs = np.transpose(joint_locs)
        joint_locs[:, 0] *= -1
        self.joint_locs = joint_locs

    def set_angles(self, angles):
        self.angles = angles
        self.get_joint_locs()
