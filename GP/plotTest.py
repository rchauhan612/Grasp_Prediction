#! /usr/bin/python

import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go

fname = "..\\HUSTdataset\\Subjects\\Subject 1 (Female)\\14 Tripod\\Sphere5\\1.txt"

r = np.genfromtxt(fname,delimiter='\t')


data = [go.Scatter(
    x = r[:, 0],
    y = r[:, 1],
    mode = 'markers'
)]

# Plot and embed in ipython notebook!
plot(data, filename='basic-scatter')
