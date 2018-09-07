import numpy as np

def get_frames(data):
    m = data.shape
    frames = []
    for i in range(1, m[0]):
        frames.append(get_single_frame(data[i, :]))
    return frames

def get_single_frame(data):
    if(data.ndim > 1):
        m = data.shape
        frame = {'data': [{'x': [data[0, 0], data[m[0]-1, 0]], 'y': [data[0, 1], data[m[0]-1, 1]], 'z': [data[0, 2], data[m[0]-1, 2]]}]}
    else:
        frame = {'data': [{'x': [data[0]], 'y': [data[1]], 'z': [data[2]]}]}
    return frame
