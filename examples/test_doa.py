import sys
sys.path.append('..')
sys.path.append('examples')
import browserinterface

import numpy as np

from micarray import MicArray
from stft import STFT
from srp import SRP
from music import MUSIC
import utils

doa = 0
nfft = 256
# microphone array positions
R_compactsix_random = np.array(
    [[51.816, 64.516, 24.13,  84.582, 44.45,  25.146],
     [36.068, 10.668, 16.002, 16.764, 10.414, 33.528],
     [ 0.0,    0.0,    0.0,    0.0,    0.0,    0.0  ]]) * 1e-3

# visualize microphone array
# ma = MicArray(R_compactsix_random)
# ma.visualize2D(plt_show=True)

def init(buffer_frames, rate, channels, volume):
    global doa, nfft, R_compactsix_random
    if channels != 6 or buffer_frames < 4096:
        browserinterface.change_config(channels=6, buffer_frames=4096)

    # create DOA object
    # doa = SRP(R_compactsix_random, fs, nfft)
    doa = MUSIC(R_compactsix_random, rate, nfft, n_grid=36)

browserinterface.register_when_new_config(init)

polar_chart = browserinterface.add_handler("Directions", 'base:polar:line', {'title': 'Direction', 'series': ['Intensity'], 'rmin': 0, 'rmax': 10, 'legend': {'from': 0, 'to': 370, 'step': 10}})

def handle_data(buffer):
    global doa, nfft

    if browserinterface.buffer_frames < 4096 or browserinterface.channels != 6:
        return

    print "Buffer received"
    # parameters
    buffer_size = browserinterface.buffer_frames
    hop_size = int(nfft/2)
    n_snapshots = int(np.floor(buffer_size/hop_size))-1

    x = buffer
    fs = browserinterface.rate

    signals = x
    X_stft = utils.compute_snapshot_spec(signals, nfft, n_snapshots, hop_size)
    doa.locate_sources(X_stft)
    to_send = doa.grid.values.tolist()
    to_send.append(to_send[0])
    polar_chart.send_data([{ 'replace': to_send }])

browserinterface.register_handle_data(handle_data)
browserinterface.start()
browserinterface.loop_callbacks()
