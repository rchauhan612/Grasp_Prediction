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
import sys
sys.path.insert(0, '../general/')
from hand_geometry_functions import *

seq_len = 20
rnn_units = 10
batch_size = 64

pred_len = 100

n_inc = 9

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

raw_data = np.load('dataset_data.npy')
raw_data_stacked = np.reshape(raw_data.astype('float32'), (raw_data.shape[0]*raw_data.shape[1], raw_data.shape[2]))

n_dims = raw_data_stacked.shape[1]

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

input_eval = tf.expand_dims(np.zeros((1, n_dims)).astype('float32'), 0)
motion_generated = []

for i in range(pred_len):
  pred = model(input_eval)
  pred_conf = tf.squeeze(pred, 0).numpy()
  input_eval = tf.expand_dims(pred_conf, 0)
  motion_generated.append(np.ravel(pred_conf))

motion_generated = np.array(motion_generated)

fig_1, ax_1 = plt.subplots()

for i in range(n_dims):
  ax_1.plot(motion_generated[:, i])

# plotting the hand motion at nine timesteps

n_rows = np.floor(np.sqrt(n_inc)).astype(int)
n_cols = np.ceil(np.sqrt(n_inc)).astype(int)
print(n_rows, n_cols)

fig_2 = plt.figure()

t_steps = np.linspace(0, pred_len-1, n_inc).astype(int)

cnt = 0
for i in range(n_rows):
    for j in range(n_cols):
        # ax.plot(np.arange(10))
        cnt += 1
        ax = fig_2.add_subplot(n_rows, n_cols, cnt, projection = '3d')
        plot_hand(calculate_joint_locs(motion_generated[t_steps[cnt-1], :]), 'black', ax=ax)

plt.show()
