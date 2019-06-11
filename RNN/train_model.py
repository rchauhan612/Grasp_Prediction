import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
import functools
from training_funcs import compute_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from itertools import zip_longest

seq_len = 20
rnn_units = 200
batch_size = 512

shift_len = 1

def split_sequences(trial, seq_len, shift_len, seqs_list = []):
    max_ind = trial.shape[0]
    for i in range(max_ind):
        if i+seq_len+shift_len > max_ind:
            break
        else:
            seqs_list.append(trial[i:i+shift_len+seq_len, :])
    return seqs_list

def split_input_target(chunk):
    return chunk[:-shift_len], chunk[seq_len:]

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

with open('train_data_padded_seperate.pickle', 'rb') as input_file:
    raw_data = pickle.load(input_file)[0]

train_seqs = []
for trial in raw_data:
    train_seqs = split_sequences(trial.astype('float32'), seq_len, shift_len, seqs_list = train_seqs)

train_seqs = np.array(train_seqs)

with open('test_data_padded_seperate.pickle', 'rb') as input_file:
    raw_data = pickle.load(input_file)[0]

test_seqs = []
for trial in raw_data:
    test_seqs = split_sequences(trial.astype('float32'), seq_len, shift_len, seqs_list = test_seqs)

test_seqs = np.array(test_seqs)

n_dims = train_seqs.shape[2]

train_seqs = tf.data.Dataset.from_tensor_slices(train_seqs)
train_dataset = train_seqs.map(split_input_target)
train_dataset = train_dataset.shuffle(1000).batch(batch_size, drop_remainder = True)

test_seqs = tf.data.Dataset.from_tensor_slices(test_seqs)
test_dataset = test_seqs.map(split_input_target)
test_dataset = test_dataset.shuffle(1000).batch(batch_size, drop_remainder = True)


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = rnn_units,
                        batch_input_shape = (batch_size, seq_len, n_dims),
                        recurrent_activation = 'sigmoid',
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    tf.keras.layers.LSTM(units = rnn_units,
                        recurrent_activation = 'sigmoid',
                        return_sequences = True,
                        dtype = 'float32'),
    tf.keras.layers.LSTM(units = rnn_units,
                        recurrent_activation = 'sigmoid',
                        return_sequences = True,
                        dtype = 'float32'),
    tf.keras.layers.Dense(n_dims, dtype = 'float32')
])

model.build((seq_len, n_dims))

print(model.summary())
input('')

EPOCHS = 20
optimizer = tf.train.AdamOptimizer(learning_rate = 0.00001)

train_hist = []
train_iter = []
test_hist = []
test_iter = []
plt.figure()

iter_cnt = 0

for epoch in range(EPOCHS):
    hidden = model.reset_states()
    print('EPOCH: {}'.format(epoch))
    # print('Training')
    for i, ((train_inp, train_target), test_data) in enumerate(tqdm(zip_longest(train_dataset, test_dataset))):
        with tf.GradientTape() as tape:
            predictions = model(train_inp)
            train_loss = compute_loss(train_target, predictions)

        grads = tape.gradient(train_loss, model.trainable_variables)

        if test_data is not None:
            (test_inp, test_target) = test_data
            predictions = model(test_inp)
            test_loss = compute_loss(test_target, predictions)
            test_hist.append(test_loss.numpy().mean())
            test_iter.append(iter_cnt)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_hist.append(train_loss.numpy().mean())
        train_iter.append(iter_cnt)

        iter_cnt += 1

    # print('Testing')
    # for i, (test_inp, test_target) in enumerate(tqdm(test_dataset)):
    #     predictions = model(test_inp)
    #     test_loss = compute_loss(test_target, predictions)
    #     test_hist.append(test_loss.numpy().mean())


    model.save_weights(checkpoint_prefix.format(epoch=epoch))

np.save(checkpoint_dir+'/loss_hist', np.array(test_hist))

plt.plot(np.array(train_iter), train_hist)
plt.plot(np.array(test_iter), test_hist)

plt.show()
