from __future__ import division
import sys
import numpy as np

sys.path.append('..')
import browserinterface

import realtimeaudio as rt

height = 200
width = 200
max = 100

buffer_size = 1024
sampling_freq = 48000
fft_size = 2*buffer_size
hop = buffer_size

def init(buffer_frames, rate, channels, volume):
    global stft

    # create STFT object
    stft = rt.transforms.STFT(fft_size, rate, hop)


def handle_data(buffer):
    global stft

    if (browserinterface.buffer_frames != buffer_size 
        or browserinterface.channels != 2):
        print("Did not receive expected audio!")
        return

    # apply stft and istft for a few windows
    stft.analysis(buffer[:,0])
    spectrum = (20 * np.log10(np.abs(stft.X[:height]))).tolist()
    spectrogram.send_data(spectrum)


"""Interface functions"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(handle_data)

spectrogram = browserinterface.add_handler(name="Heat Map", type='base:spectrogram',
        parameters={'width': width, 'height': height, 'min': 0, 'max': 150, 'delta_freq': sampling_freq / fft_size})

"""START"""
browserinterface.change_config(channels=2, buffer_frames=buffer_size, 
    volume=80, rate=sampling_freq)
browserinterface.start()
browserinterface.loop_callbacks()
