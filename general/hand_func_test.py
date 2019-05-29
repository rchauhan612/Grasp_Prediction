from hand_geometry_functions import*

test_angles = np.zeros(16)

joint_locs = calculate_joint_locs(test_angles)
plot_hand(joint_locs)
