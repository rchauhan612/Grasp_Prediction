# %% import everything
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config = config)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

# %%
rnn_units = 200
batch_size = 20
n_dims = 16

test_groups = False
num_cat = 8 if test_groups else 33
train_split = 0.66

# %%

def gather_data(filename = 'all_data_seperate.pickle'):
    with open(filename, 'rb') as input_file:
        raw_data = pickle.load(input_file)

    return raw_data[0], raw_data[1] # traj, group + grasp

def create_sequences(traj_data, labels, num_seqs = 5, train_split = 0.66):
    seq_list = []
    label_list = []
    for trial, label in zip(traj_data, labels):
        inds = np.linspace(0, len(trial), num_seqs + 1)[1:].astype('int')
        for ind in inds:
            seq_list.append(trial[:ind])
            temp = np.zeros(num_cat)
            temp[label-1] = 1
            label_list.append(temp)

    print('Number of sequences: {}'.format(len(seq_list)))

    max_len = max([seq.shape[0] for seq in seq_list])
    seq_list = [np.vstack((np.zeros((max_len - seq.shape[0] , n_dims)), seq)) for seq in seq_list]

    train_size = int(train_split * len(seq_list))

    data = tf.data.Dataset.from_tensor_slices((seq_list, label_list))
    data = data.shuffle(len(seq_list)+1)

    train_data = data.take(train_size).batch(batch_size, drop_remainder = True)
    test_data = data.skip(train_size)

    return train_data, test_data, len(seq_list)

# %%

def cce_loss(pred, truth, num_cat):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(pred, truth)

# %%

def pred(model, traj, label):
    pred_class = np.amax(model(traj)) + 1
    print('True class: {}\t Predicted class: {}'.format(label, pred_class))

# %%

print('Collecting data...')
trajs, labels = gather_data()

if test_groups:
    labels = [l[0] for l in labels]
else:
    labels = [l[1] for l in labels]

print('Forming dataset...')
train_data, test_data, num_seqs = create_sequences(trajs, labels, num_seqs = 1)

# %%

print('Creating model...')

model = tf.keras.Sequential([
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        batch_input_shape = (batch_size, None, n_dims),
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        return_sequences = True,
                        dtype = 'float32'),
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        return_sequences = False,
                        dtype = 'float32'),
    tf.keras.layers.Dense(num_cat, activation=tf.nn.softmax, dtype = 'float32')
])

model.build((None, n_dims))
print(model.summary())

# %%

print('Training model...')
EPOCHS = 100
learning_rate = 0.000001
# optimizer = tf.train.AdagradOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer(learning_rate)
optimizer = tf.train.RMSPropOptimizer(learning_rate)
loss_hist = []
for epoch in range(EPOCHS):
    print('Epoch: {}/{}'.format(epoch+1, EPOCHS))
    # hidden = model.reset_states()
    for _, dat in tqdm(enumerate(train_data), total = int(train_split * num_seqs / batch_size)):
        traj = tf.cast(dat[0], dtype = tf.float32)
        label = dat[1]
        with tf.GradientTape() as tape:
            current_loss = cce_loss(model(traj),
                                    label,
                                    num_cat)

        grads = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_hist.append(current_loss)

# %%
plt.plot(loss_hist)
plt.show()

# %%

print(len(loss_hist))
print(loss_hist[0])
