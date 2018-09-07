class Finger:
    def __init__(self, link_lengths):
        self.lengths = link_lengths
        self.th = np.array([0, 0, 0])

    def set_angles(self, angles):
        self.angles = angles

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
        return joint_locs
