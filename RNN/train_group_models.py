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

checkpoint_dir = './group_training_checkpoints'

for g in range(1, 9):
    checkpoint_prefixes = os.path.join(checkpoint_dir, "group_{group}_ckpt_{epoch}".format(group = g))

    train_data = np.load('group_data/{}_data.npy'.format(g)).astype('float32')
    n_dims = train_data.shape[1]

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)

    train_seqs = train_dataset.batch(seq_len+shift_len, drop_remainder = True)

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
        tf.keras.layers.Dense(n_dims, dtype = 'float32')
    ])

    model.build((seq_len, n_dims))

    EPOCHS = 5
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
