"""
Perform frequency domain analysis with STFT (50% overlap)
"""

from __future__ import division
import numpy as np

import browserinterface
import algorithms as rt

"""Spectrogram parameters"""
buffer_size = 1024
max_freq = 4000; max_val = 200; width = 200
transform = 'fftw' # 'numpy', 'mlk', 'fftw'

"""Read hardware config from file"""
try:
    import json
    with open('./hardware_config.json', 'r') as config_file:
        config = json.load(config_file)
        config_file.close()
    sampling_freq = config['sampling_frequency']
except:
    # default when no hw config file is present
    sampling_freq = 44100

fft_size = 2*buffer_size; hop = buffer_size
max_freq = min(sampling_freq/2, max_freq);
height = int(np.ceil(float(max_freq)/sampling_freq*fft_size))
num_channels=2


def init(buffer_frames, rate, channels, volume):
    global stft

    # create STFT object
    stft = rt.transforms.STFT(fft_size, rate, hop, transform=transform)


def handle_data(audio):
    global stft

    # check for correct audio shape
    if audio.shape != (buffer_size, num_channels):
        print("Did not receive expected audio!")
        return

    # apply stft and istft for a few windows
    stft.analysis(audio[:,0])
    spectrum = (20 * np.log10(np.abs(stft.X[:height]))).tolist()
    spectrogram.send_data(spectrum)


"""Interface functions"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(handle_data)

spectrogram = browserinterface.add_handler(name="Heat Map", type='base:spectrogram',
        parameters={'width': width, 'height': height, 'min': 0, 'max': max_val, 'delta_freq': sampling_freq / fft_size})

"""START"""
browserinterface.start()
browserinterface.change_config(channels=num_channels, buffer_frames=buffer_size, volume=80, rate=sampling_freq)
browserinterface.loop_callbacks()
