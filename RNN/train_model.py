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

seq_len = 20
rnn_units = 10
batch_size = 64

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

raw_data = np.load('dataset_data.npy')
raw_data_stacked = np.reshape(raw_data.astype('float32'), (raw_data.shape[0]*raw_data.shape[1], raw_data.shape[2]))

n_dims = raw_data_stacked.shape[1]

movement_dataset = tf.data.Dataset.from_tensor_slices(raw_data_stacked)

seqs = movement_dataset.batch(seq_len+1, drop_remainder = True)

dataset = seqs.map(split_input_target)

dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder = True)

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

history = []
plt.figure()

for epoch in range(EPOCHS):
    print('EPOCH:', epoch)
    hidden = model.reset_states()
    bar = tqdm(total = 1215, position = 0)
    for i, (inp, target) in enumerate(dataset):
        bar.update(1)
        with tf.GradientTape() as tape:
            predictions = model(inp)
            loss = compute_loss(target, predictions)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        history.append(loss.numpy().mean())

    model.save_weights(checkpoint_prefix.format(epoch=epoch))

plt.plot(history)
plt.show()
