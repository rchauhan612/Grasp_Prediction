import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
import functools
from IPython import display as ipythondisplay
from training_funcs import compute_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

seq_len = 10
rnn_units = 500
batch_size = 128

shift_len = 1

def split_input_target(chunk):
    return chunk[:-shift_len], chunk[shift_len:]

checkpoint_dir = './class_training_checkpoints/'

with open('train_data_seperate.pickle', 'rb') as input_file:
    raw_data = pickle.load(input_file)

motion_data = raw_data[0]
motion_data = np.concatenate(motion_data)
group_nums = raw_data[1]

n_dims = motion_data.shape[1]

train_dataset = tf.data.Dataset.from_tensor_slices(motion_data)
train_seqs = train_dataset.batch(seq_len+shift_len, drop_remainder = True)
train_dataset = train_seqs.map(split_input_target)
train_dataset = train_dataset.shuffle(1000).batch(batch_size, drop_remainder = True)

fig = plt.figure()

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = rnn_units,
                        batch_input_shape = (batch_size, seq_len, n_dims),
                        recurrent_activation = 'sigmoid',
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    tf.keras.layers.Dense(8, activation = tf.nn.softmax)
])

model.build((seq_len, n_dims))

EPOCHS = 5
optimizer = tf.train.AdamOptimizer()

train_hist = []

for epoch in range(EPOCHS):
    hidden = model.reset_states()
    for i, (train_inp, train_target) in enumerate(tqdm(train_dataset)):

        with tf.GradientTape() as tape:

            predictions = model(train_inp)
            train_loss = compute_loss(train_target, predictions)

        grads = tape.gradient(train_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_hist.append(train_loss.numpy().mean())

model.save_weights(checkpoint_dir + 'model.h5'.format(group = g))

plot.plot(train_hist)

plt.show()
