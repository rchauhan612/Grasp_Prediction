import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
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
seed_percent = .30

test_trial_num = 4

n_inc = 9

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

with open('test_data_seperate.pickle', 'rb') as input_file:
    raw_data = pickle.load(input_file)
test_trial = raw_data[test_trial_num]
test_conf = raw_data[test_trial_num][-1]
# print(test_trial)
# print(raw_data.shape)

# raw_data_stacked = np.reshape(test_trial.astype('float32'), (test_trial.shape[0]*test_trial.shape[1], test_trial.shape[2]))
n_dims = test_trial.shape[1]
seed_len = np.floor(test_trial.shape[0] * seed_percent).astype(int)
pred_len = np.floor(test_trial.shape[0] * pred_percent).astype(int)

model = tf.keras.Sequential([tf.keras.layers.LSTM(units = rnn_units,
                                                  input_shape = (seq_len, n_dims),
                                                  recurrent_activation = 'sigmoid',
                                                  return_sequences = True,
                                                  recurrent_initializer = 'glorot_uniform',
                                                  stateful = False,
                                                  dtype = 'float32'),
                              tf.keras.layers.Dense(n_dims, dtype = 'float32')
])

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.reset_states()

# input_eval = tf.expand_dims(np.zeros((1, n_dims)).astype('float32'), 0)
input_eval = test_trial[:seed_len, :].astype('float32')

motion_generated = input_eval

for i in tqdm(range(pred_len-seed_len)):
  pred = model(tf.expand_dims(motion_generated, 0))
  motion_generated = np.vstack((motion_generated, np.squeeze(pred)[-1]))

motion_generated = np.array(motion_generated)

plt.ion()

fig_1, ax_1 = plt.subplots()

for i in range(n_dims):
  ax_1.plot(np.linspace(0, 200, len(motion_generated[:, i])), motion_generated[:, i])

# plotting the hand motion at nine timesteps

n_rows = np.floor(np.sqrt(n_inc)).astype(int)
n_cols = np.ceil(np.sqrt(n_inc)).astype(int)
# print(n_rows, n_cols)

fig_2 = plt.figure()

t_steps = np.linspace(0, pred_len-seed_len-1, n_inc).astype(int)

# print(motion_generated.shape)

cnt = 0
for i in range(n_rows):
    for j in range(n_cols):
        # ax.plot(np.arange(10))
        cnt += 1
        ax = fig_2.add_subplot(n_rows, n_cols, cnt, projection = '3d')
        plot_hand(calculate_joint_locs(test_conf.copy()), 'black', ax=ax)
        if t_steps[cnt-1] <= seed_len:
            plot_hand(calculate_joint_locs(motion_generated[t_steps[cnt-1], :].copy()), 'green', ax=ax)
        else:
            plot_hand(calculate_joint_locs(motion_generated[t_steps[cnt-1], :].copy()), 'red', ax=ax)

        ax.set_title('{:.2f}%'.format(200*cnt/n_inc))
        ax.set_axis_off()

plt.show()
plt.pause(0.0001)
input('')
