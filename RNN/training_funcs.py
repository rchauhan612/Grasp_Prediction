import tensorflow as tf

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

def compute_loss(labels, logits):
    return tf.keras.losses.MSE(labels, logits)
