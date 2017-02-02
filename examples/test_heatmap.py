import sys
import numpy as np

sys.path.append('..')
import browserinterface


def handle_data(buffer):
    # data = [[2, 3, 4, 0], [1, 0, 0, 1],[3, 4, 4, 1]]
    data = np.random.rand(1000,100).tolist()
    heatmap.send_data(data)

"""Interface functions"""
browserinterface.register_handle_data(handle_data)
heatmap = browserinterface.add_handler(name="Heat Map", type='base:heatmap',
    parameters={'min': 0, 'max': 1})

"""START"""
browserinterface.start()
browserinterface.loop_callbacks()