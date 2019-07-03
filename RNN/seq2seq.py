# %% import everything
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config = config)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import pickle
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from make_conf_matrix import plot_confusion_matrix

# %%
rnn_units = 300
batch_size = 128
n_dims = 16

# test_set = [2, 14, 20, 17, 10, 22, 15, 31]
test_set = range(1, 34)

test_groups = False
num_cat = 8 if test_groups else len(test_set)
train_split = 0.66

PAD_conf = 99 * np.ones((1, n_dims))
EOL_conf = -99 * np.ones((1, n_dims))
BOS_conf = -44 * np.ones((1, n_dims))

# %%

def gather_data(filename = 'all_data_seperate.pickle'):
    with open(filename, 'rb') as input_file:
        raw_data = pickle.load(input_file)

    return raw_data[0], raw_data[1] # traj, group + grasp

def normalize_data(data, scaler = None):

    if scaler is None:
        scaler = StandardScaler(with_std = False)
        stacked_data = np.concatenate(data)
        scaler.fit(stacked_data)

    transformed_data = data.copy()
    for i, dat in enumerate(transformed_data):
        transformed_data[i] = scaler.transform(dat)

    return scaler, transformed_data

def create_sequences(traj_data, labels, num_seqs = 5, train_split = 0.66, which_seqs = None,
                        include_set = None):
    in_seq_list = []
    out_seq_list = []
    label_list = []

    test_set_norm = np.arange(len(test_set)) + 1

    for trial, label in zip(traj_data, labels):
        if include_set is None:
            include_set = np.arange(num_cat) + 1
            include_set = include_set.tolist()
        if label in include_set:
            label = include_set.index(label) + 1
            if which_seqs == None:
                inds = np.linspace(0, len(trial), num_seqs + 1)[1:].astype('int')
            else:
                inds = np.linspace(0, len(trial), num_seqs + 1)[1:which_seqs+1].astype('int')

            for ind in inds:
                in_seq_list.append(trial[:ind])
                out_seq_list.append(trial[ind:])
                temp = np.zeros(num_cat)
                temp[label-1] = 1
                label_list.append(temp)
                # print(temp)

    print('Number of sequences: {}'.format(len(in_seq_list)))

    max_in_len = max([seq.shape[0] for seq in in_seq_list])
    in_seq_list = [np.vstack((seq, np.repeat(PAD_conf, max_in_len - seq.shape[0], axis = 0))) for seq in in_seq_list]

    max_out_len = max([seq.shape[0] for seq in out_seq_list])
    out_seq_list = [np.vstack((BOS_conf, seq, np.repeat(EOL_conf, max_out_len - seq.shape[0], axis = 0))) for seq in out_seq_list]

    train_size = int(train_split * len(in_seq_list))

    data = tf.data.Dataset.from_tensor_slices((in_seq_list, out_seq_list, label_list))
    data = data.shuffle(len(in_seq_list)+1)

    train_data = data.take(train_size).batch(batch_size, drop_remainder = True)
    test_data = data.skip(train_size)

    return train_data, test_data, (len(in_seq_list), train_size, len(in_seq_list)-train_size)

# %%

def cce_loss(pred, truth):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(pred, truth)

def calc_acc(pred, truth):
    pred = pred.numpy()
    truth = truth.numpy()

    acc = []
    for p, t in zip(pred, truth):
        acc.append(np.argmax(p) == np.argmax(t))

    return np.mean(acc)
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

# %%

print('Normalizing data...')
normalizing_scaler, trajs = normalize_data(trajs)

# %%

print('Forming dataset...')
train_data, test_data, (num_seqs, num_train_seq, num_test_seq) = create_sequences(trajs, labels, num_seqs = 1, which_seqs = None, include_set = test_set)
#
# # %%
#
# if not os.path.exists('datasets'):
#     os.makedirs('datasets')
# with open('./datasets/seq2seq_train_data.pickle', 'wb') as output_file:
#     pickle.dump(train_data.numpy(), output_file)
# with open('./datasets/seq2seq_test_data.pickle', 'wb') as output_file:
#     pickle.dump(test_data.numpy(), output_file)

# %%

print('Creating model...')

model = tf.keras.Sequential([
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        batch_input_shape = (batch_size, None, n_dims),
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    # tf.keras.layers.Dropout(rate = .8),
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        return_sequences = True,
                        dtype = 'float32'),
    # tf.keras.layers.Dropout(rate = .8),
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        return_sequences = False,
                        dtype = 'float32'),
    # tf.keras.layers.Dropout(rate = .8),
    tf.keras.layers.Dense(num_cat, activation=tf.nn.softmax, dtype = 'float32')
])

model.build((None, n_dims))
print(model.summary())

# %%

print('Training model...')

checkpoint_dir = './training_checkpoints_class'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

EPOCHS = 10000
learning_rate = 0.00001
# optimizer = tf.train.AdagradOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.RMSPropOptimizer(learning_rate, centered = True)
loss_hist = []
acc_hist = []
# for epoch in range(EPOCHS):
this_epoch_acc = 0
epoch = 0
while np.mean(this_epoch_acc) < .95 and epoch < EPOCHS:
    # print('Epoch: {}/{}'.format(epoch+1, EPOCHS))
    this_epoch_loss = []
    this_epoch_acc = []
    hidden = model.reset_states()
    for _, dat in tqdm(enumerate(train_data), total = int(train_split * num_seqs / batch_size)):
        traj = tf.cast(dat[0], dtype = tf.float32)
        label = dat[1]

        with tf.GradientTape() as tape:
            pred = model(traj)
            current_loss = cce_loss(pred, label)
            current_acc = calc_acc(pred, label)

        grads = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        this_epoch_loss.append(current_loss)
        this_epoch_acc.append(current_acc)

    loss_hist.append(np.mean(this_epoch_loss))
    acc_hist.append(np.mean(this_epoch_acc))

    if epoch % 50 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    epoch += 1

    print('Epoch {0}/{1} loss: {2}\tAcc: {3:.3f}%'.format(epoch+1, EPOCHS, np.mean(this_epoch_loss), 100*np.mean(this_epoch_acc)))

# %%

plt.plot(loss_hist)
plt.xlabel('Epochs')
plt.ylabel('Loss (CCE)')
plt.show()

# %%

model = tf.keras.Sequential([
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        batch_input_shape = (1, None, n_dims),
                        return_sequences = True,
                        recurrent_initializer = 'glorot_uniform',
                        stateful = True,
                        dtype = 'float32'),
    # tf.keras.layers.Dropout(rate = .8),
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        return_sequences = True,
                        dtype = 'float32'),
    # tf.keras.layers.Dropout(rate = .8),
    tf.keras.layers.CuDNNLSTM(units = rnn_units,
                        return_sequences = False,
                        dtype = 'float32'),
    # tf.keras.layers.Dropout(rate = .8),
    tf.keras.layers.Dense(num_cat, activation=tf.nn.softmax, dtype = 'float32')
])

model.build((None, n_dims))
print(model.summary())

checkpoint_dir = './training_checkpoints_class'
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.reset_states()

# %%

test_res = [[], []]
num_splits = 7
for i, dat in tqdm(enumerate(test_data)):
    traj = tf.expand_dims(tf.cast(dat[0], dtype = tf.float32), axis = 0)
    label = dat[1]
    print(label)
    sparse_label = np.argmax(label.numpy())
    traj = traj.numpy()
    num_mes = traj.shape[1]
    for j in range(1, num_splits+1):
        test_res[0].append(sparse_label)

        sub_traj = traj[:, :int(num_mes * j / num_splits), :]
        predictions = model(sub_traj)
        sparse_pred = np.argmax(predictions.numpy())
        test_res[1].append(sparse_pred)

# %%

print(num_test_seq)
print(test_res[1][:100])

# %%

test_acc = [[] for i in range(num_cat)]
for label, pred in zip(test_res[0], test_res[1]):
    test_acc[label].append(int(label == pred))

test_acc = [np.mean(gr) for gr in test_acc]

print('Total Accuracy: {}'.format(np.mean(test_acc)))

plt.figure()
plt.bar(np.arange(1, 34), np.array(test_acc))
plt.xlabel('Grasp')
plt.xticks(np.linspace(1, 33, 10).astype('int'))
plt.ylabel('Accuracy (%)')

# %%

plot_confusion_matrix(test_res[0], test_res[1], classes = 1+np.arange(33), normalize = True)
plt.show()
