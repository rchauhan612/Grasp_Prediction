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

seq_len = 10
rnn_units = 500
batch_size = 32

shift_len = 5

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# raw_data = np.load('dataset_data.npy')
# raw_data_stacked = np.reshape(raw_data.astype('float32'), (raw_data.shape[0]*raw_data.shape[1], raw_data.shape[2]))
train_data = np.load('training_data.npy').astype('float32')
test_data = np.load('testing_data.npy').astype('float32')

n_dims = train_data.shape[1]

train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

train_seqs = train_dataset.batch(seq_len+shift_len, drop_remainder = True)
test_seqs = test_dataset.batch(seq_len+shift_len, drop_remainder = True)

train_dataset = train_seqs.map(split_input_target)
test_dataset = test_seqs.map(split_input_target)

train_dataset = train_dataset.shuffle(1000).batch(batch_size, drop_remainder = True)
test_dataset = test_dataset.shuffle(1000).batch(batch_size, drop_remainder = True)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = rnn_units,
                        batch_input_shape = (batch_size, seq_len, n_dims),
                        recurrent_activation = 'sigmoid',
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
            # motion_generated = train_inp.numpy().copy()
            # for j in range(shift_len):
            #     pred = model(tf.expand_dims(train_inp, 0))
            #     motion_generated = np.vstack((motion_generated, np.squeeze(pred)[-1]))
            #
            # motion_generated = np.array(motion_generated)
            # input('')
            # train_loss = compute_loss(train_target, motion_generated[-shift_len:])

            predictions = model(train_inp)
            train_loss = compute_loss(train_target, predictions)

        grads = tape.gradient(train_loss, model.trainable_variables)

        # predictions = model(test_inp)
        # test_loss = compute_loss(test_target_0, predictions)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_hist.append(train_loss.numpy().mean())
        # test_hist.append(test_loss.numpy().mean())

    model.save_weights(checkpoint_prefix.format(epoch=epoch))

plt.plot(train_hist)
# plt.plot(test_hist)
plt.show()
