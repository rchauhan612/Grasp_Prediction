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

seq_len = 5
rnn_units = 100
batch_size = 16

shift_len = 1

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

train_data = np.load('training_data.npy').astype('float32')

n_dims = train_data.shape[1]

train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_seqs = train_dataset.batch(seq_len+shift_len, drop_remainder = True)
train_dataset = train_seqs.map(split_input_target)
train_dataset = train_dataset.shuffle(1000).batch(batch_size, drop_remainder = True)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = rnn_units,
                        batch_input_shape = (batch_size, seq_len, n_dims),
                        recurrent_activation = 'relu',
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    tf.keras.layers.LSTM(units = rnn_units,
                        batch_input_shape = (batch_size, seq_len, n_dims),
                        recurrent_activation = 'relu',
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    tf.keras.layers.LSTM(units = rnn_units,
                        batch_input_shape = (batch_size, seq_len, n_dims),
                        recurrent_activation = 'relu',
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    tf.keras.layers.Dense(n_dims, dtype = 'float32')
])

model.build((seq_len, n_dims))

EPOCHS = 7
optimizer = tf.train.AdamOptimizer()

train_hist = []
test_hist = []
plt.figure()

for epoch in range(EPOCHS):
    hidden = model.reset_states()
    for i, ((test_inp, test_target), (train_inp, train_target)) in enumerate(tqdm(zip(test_dataset, train_dataset))):

        with tf.GradientTape() as tape:

            predictions = model(train_inp)
            train_loss = compute_loss(train_target, predictions)

        grads = tape.gradient(train_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_hist.append(train_loss.numpy().mean())

    model.save_weights(checkpoint_prefix.format(epoch=epoch))

plt.plot(train_hist)
plt.show()
