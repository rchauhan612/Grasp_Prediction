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
import pickle


seq_len = 5
rnn_units = 100
batch_size = 32

shift_len = 3


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

with open('train_data_seperate.pickle', 'rb') as input_file:
    train_data = pickle.load(input_file)

n_dims = train_data[0].shape[1]

pad = 0*np.ones((1, n_dims))

for i, trial in enumerate(train_data):
    # trial = np.vstack((trial, pad)) # pad the trials with data that the network will never see again
    train_data[i] = trial.astype('float32')

train_data = np.concatenate(train_data)
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
train_seqs = train_dataset.batch(seq_len+shift_len, drop_remainder = True)
# print('here')
# print(train_seqs.output_shapes)
# input('')
# train_seqs = train_seqs.filter(lambda x: any(np.array_equal(mes, pad) for mes in x.numpy()))
train_dataset = train_seqs.map(split_input_target)
train_dataset = train_dataset.shuffle(10000).batch(batch_size, drop_remainder = True)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = rnn_units,
                        batch_input_shape = (batch_size, seq_len, n_dims),
                        recurrent_activation = 'tanh',
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    tf.keras.layers.Dense(n_dims, dtype = 'float32')
])

model.build((seq_len, n_dims))

EPOCHS = 10
optimizer = tf.train.AdamOptimizer()

train_hist = []
test_hist = []
plt.figure()

for epoch in range(EPOCHS):
    hidden = model.reset_states()
    for (train_inp, train_target) in tqdm(train_dataset):
        with tf.GradientTape() as tape:
            predictions = model(train_inp)
            train_loss = compute_loss(train_target, predictions)

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_hist.append(train_loss.numpy().mean())

    model.save_weights(checkpoint_prefix.format(epoch=epoch))

plt.plot(train_hist)
plt.show()
