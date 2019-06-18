import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config = config)

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

rnn_units = 200
batch_size = 512
n_dims = 16

test_groups = True
num_cat = 8 if test_groups else 33

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
            seq_list.append(trial[:ind].astype('float32'))
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

    return train_data, test_data

def cce_loss(pred, truth, num_cat):
    cce = tf.keras.losses.CategoricalCrossentropy()

    # truth_arr = np.zeros(batch_size, num_cat)
    # truth_arr[truth-1] = 1
    return cce(pred, truth)

def train_model(model, data, EPOCHS = 5, learning_rate = 0.001):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    loss_hist = []
    for epoch in range(EPOCHS):
        print('Epoch: {}/{}'.format(epoch+1, EPOCHS))
        hidden = model.reset_states()
        for dat in tqdm(data):
            traj = dat[0]
            label = dat[1]
            with tf.GradientTape() as tape:
                current_loss = cce_loss(model(tf.cast(traj, dtype = tf.float32)), label, num_cat)

            grads = tape.gradient(current_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_hist

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.CuDNNLSTM(units = rnn_units,
                            batch_input_shape = (batch_size, None, n_dims),
                            return_sequences = False,
                            recurrent_initializer = 'glorot_uniform',
                            stateful = True,
                            dtype = 'float32'),
        tf.keras.layers.Dense(num_cat, activation=tf.nn.softmax, dtype = 'float32')
    ])

    model.build((None, n_dims))
    return model

def pred(model, traj, label):
    pred_class = np.amax(model(traj)) + 1
    print('True class: {}\t Predicted class: {}'.format(label, pred_class))

if __name__ == '__main__':
    trajs, labels = gather_data()

    if test_groups:
        labels = [l[0] for l in labels]
    else:
        labels = [l[1] for l in labels]

    print('Forming dataset...')
    train_data, test_data = create_sequences(trajs, labels)

    print('Creating model...')
    model = create_model()
    print(model.summary())
    input('Done')

    print('Training model...')
    loss_hist = train_model(model, train_data)
    input('Done')

    plt.plot(loss_hist)
