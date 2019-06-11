import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config = config)
import numpy as np
import os
import time
import functools
from training_funcs import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import sys
sys.path.insert(0, '../general/')
from hand_geometry_functions import *

seq_len = 20
rnn_units = 500

pred_percent = 2
seed_percent = .25

n_inc = 9

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

with open('test_data_seperate.pickle', 'rb') as input_file:
    data = pickle.load(input_file)
    raw_data = data[0]
    labels = data[1]

test_trial_num = np.random.randint(len(raw_data))
test_trial = raw_data[test_trial_num]
test_conf = raw_data[test_trial_num][-1]
test_trial_group_num = labels[test_trial_num]

n_dims = test_trial.shape[1]
seed_len = np.floor(test_trial.shape[0] * seed_percent).astype(int)
pred_len = np.floor(test_trial.shape[0] * pred_percent).astype(int)

test_trial_extended = np.vstack((test_trial, np.tile(test_trial[[-1], :], (pred_len - test_trial.shape[0], 1))))

print('Test trial number: {}\t Grasp Number: {}'.format(test_trial_num, test_trial_group_num))

model = tf.keras.Sequential([
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        batch_input_shape = (1, seq_len, n_dims),
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        return_sequences = True,
                        dtype = 'float32'),
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        return_sequences = True,
                        dtype = 'float32'),
    tf.keras.layers.Dense(n_dims, dtype = 'float32')
])

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.reset_states()

# input_eval = tf.expand_dims(np.zeros((1, n_dims)).astype('float32'), 0)

motion_generated = []
for i in tqdm(range(1, test_trial_extended.shape[0])):
    if i < seq_len:
        # pred = model(tf.expand_dims(test_trial_extended[:i+1, :].astype('float32'), 0))
        pred = np.expand_dims(test_trial_extended[:i+1, :].astype('float32'), 0)
    else:
        pred = model(tf.expand_dims(test_trial_extended[i+1-seq_len:i+1, :].astype('float32'), 0))

    # print(pred.shape)
    motion_generated.append(np.squeeze(pred)[-1])

motion_generated = np.array(motion_generated)

plt.ion()

fig_1, ax_1 = plt.subplots()

for i in range(n_dims):
    ax_1.plot(np.linspace(0, pred_len, test_trial_extended.shape[0]), test_trial_extended[:, i], color = 'C0')
    ax_1.plot(np.linspace(1, pred_len, test_trial_extended.shape[0]-1), motion_generated[:, i], color = 'C1')

# plotting the hand motion at nine timesteps

n_rows = np.floor(np.sqrt(n_inc)).astype(int)
n_cols = np.ceil(np.sqrt(n_inc)).astype(int)
# print(n_rows, n_cols)

fig_2 = plt.figure()

t_steps = np.linspace(0, motion_generated.shape[0]-1, n_inc).astype(int)

cnt = 0

print(motion_generated.shape, t_steps)
for i in range(n_rows):
    for j in range(n_cols):
        # ax.plot(np.arange(10))
        cnt += 1
        ax = fig_2.add_subplot(n_rows, n_cols, cnt, projection = '3d')
        plot_hand(calculate_joint_locs(test_conf.copy()), 'black', ax=ax)
        plot_hand(calculate_joint_locs(motion_generated[t_steps[cnt-1], :].copy()), 'red', ax=ax)

        # print(motion_generated[t_steps[cnt-1], :].copy())

        ax.set_title('{:.2f}%'.format(200*cnt/n_inc))
        ax.set_axis_off()

fig_2.suptitle('Test trial number: {}\t Grasp Number: {}'.format(test_trial_num, test_trial_group_num))

plt.show()
plt.pause(0.0001)
input('')
