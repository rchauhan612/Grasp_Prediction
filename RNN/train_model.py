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

raw_data = np.load('dataset_data.npy')
raw_data_stacked = np.reshape(raw_data.astype('float32'), (raw_data.shape[0]*raw_data.shape[1], raw_data.shape[2]))

n_dims = raw_data_stacked.shape[1]

movement_dataset = tf.data.Dataset.from_tensor_slices(raw_data_stacked)

# print(movement_dataset)

seqs = movement_dataset.batch(seq_len+1, drop_remainder = True)

# print(seqs)

dataset = seqs.map(split_input_target)

# print(dataset)

dataset = dataset.shuffle(1000).batch(batch_size, drop_remainder = True)

# print(dataset)

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

print(model.summary())
#
# for input_example_batch, target_example_batch in dataset.take(1):
#   example_batch_predictions = model(input_example_batch)
#   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


EPOCHS = 5
optimizer = tf.train.AdamOptimizer()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

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

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = rnn_units,
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

for i in range(100):
    pred = model(input_eval)
    pred_conf = tf.squeeze(pred, 0).numpy()
    input_eval = tf.expand_dims(pred_conf, 0)
    motion_generated.append(np.ravel(pred_conf))

motion_generated = np.array(motion_generated)

plt.figure()

print(motion_generated)

for i in range(n_dims):
    plt.plot(motion_generated[:, i])

plt.show()
