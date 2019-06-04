import tensorflow as tf

def split_input_target(chunk):
    return chunk[:-5], chunk[5:]

def compute_loss(labels, logits):
    return tf.keras.losses.MSE(labels, logits)
