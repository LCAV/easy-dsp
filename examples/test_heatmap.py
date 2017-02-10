from __future__ import division
import sys
import numpy as np

sys.path.append('..')
import browserinterface

height = 50
width = 200
max = 100

i = [0]

def handle_data(buffer):
    a = np.sin(2 * np.pi * (i[0] * 1.) / width) * max
    data = np.floor((a * np.random.rand(100))).tolist()
    i[0] = (i[0] + 1) % width
    heatmap.send_data(data)

"""Interface functions"""
browserinterface.register_handle_data(handle_data)
heatmap = browserinterface.add_handler(name="Heat Map", type='base:spectrogram',
        parameters={'width': width, 'height': height, 'min': -max, 'max': max})

"""START"""
browserinterface.start()
browserinterface.loop_callbacks()
