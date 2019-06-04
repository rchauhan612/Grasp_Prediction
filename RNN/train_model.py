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

seq_len = 20
rnn_units = 100
batch_size = 256

shift_len = 1

def split_sequences(trial, seq_len, shift_len, seqs_list = []):
    max_ind = trial.shape[0]
    for i in range(max_ind):
        if i+seq_len+shift_len > max_ind:
            break
        else:
            seqs.append(trial[i:i+shift_len+seq_len, :])
    return seqs_list

def split_input_target(chunk):
    return chunk[:-shift_len], chunk[shift_len:]

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

with open('train_data_seperate.pickle', 'rb') as input_file:
    raw_data = pickle.load(input_file)[0]

seqs = []
for trial in raw_data:
    seqs = split_sequences(trial.astype('float32'), seq_len, shift_len, seqs_list = seqs)

seqs = np.array(seqs)

print(seqs.shape)

n_dims = seqs.shape[2]

train_seqs = tf.data.Dataset.from_tensor_slices(seqs)
train_dataset = train_seqs.map(split_input_target)
train_dataset = train_dataset.shuffle(1000).batch(batch_size, drop_remainder = True)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = rnn_units,
                        batch_input_shape = (batch_size, seq_len, n_dims),
                        recurrent_activation = 'sigmoid',
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    # tf.keras.layers.LSTM(units = rnn_units,
    #                     recurrent_activation = 'sigmoid',
    #                     return_sequences = True,
    #                     dtype = 'float32'),
    # tf.keras.layers.LSTM(units = rnn_units,
    #                     recurrent_activation = 'sigmoid',
    #                     return_sequences = True,
    #                     dtype = 'float32'),
    tf.keras.layers.Dense(n_dims, dtype = 'float32')
])

model.build((seq_len, n_dims))

EPOCHS = 1
optimizer = tf.train.AdamOptimizer()

train_hist = []
plt.figure()

for epoch in range(EPOCHS):
    hidden = model.reset_states()
    for i, (train_inp, train_target) in enumerate(tqdm(train_dataset)):

        with tf.GradientTape() as tape:

            predictions = model(train_inp)
            train_loss = compute_loss(train_target, predictions)

        grads = tape.gradient(train_loss, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_hist.append(train_loss.numpy().mean())

    model.save_weights(checkpoint_prefix.format(epoch=epoch))

plt.plot(train_hist)
plt.show()
