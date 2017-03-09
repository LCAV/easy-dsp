"""
Perform frequency domain analysis with STFT (50% overlap)
"""

from __future__ import division, print_function
import sys
import numpy as np
import time

import browserinterface
import algorithms as rt

"""Spectrogram parameters"""
buffer_size = 1024
max_freq = 3000
min_val = 60
max_val = 120
width = 200
transform = 'mkl' # 'numpy', 'mlk', 'fftw'

"""Read hardware config from file"""
try:
    import json
    with open('./hardware_config.json', 'r') as config_file:
        config = json.load(config_file)
        config_file.close()
    sampling_freq = config['sampling_frequency']
    led_ring_address = config['led_ring_address']
except:
    # default when no hw config file is present
    sampling_freq = 44100
    led_ring_address = '/dev/cu.usbmodem1411'

fft_size = 2*buffer_size; hop = buffer_size
max_freq = min(sampling_freq/2, max_freq);
height = int(np.ceil(float(max_freq)/sampling_freq*fft_size))
num_channels=2

"""Check for LED Ring"""
try:
    import matplotlib.cm as cm
    led_ring = rt.neopixels.NeoPixels(usb_port=led_ring_address,
        colormap=cm.summer, vrange=[min_val, max_val])
    print("LED ring ready to use!")
except:
    print("No LED ring available...")
    led_ring = False


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
    spectrum = (20 * np.log10(np.abs(stft.X[:height])))
    spectrogram.send_data(spectrum.tolist())


    if led_ring:
        numpix = led_ring.num_pixels // 2
        ma_len = int(np.floor(spectrum.shape[0] / numpix))
        spec = np.convolve(np.ones(ma_len) / ma_len, spectrum)[ma_len//2::ma_len]
        spec2 = np.concatenate((spec[:numpix], spec[numpix::-1]))
        led_ring.lightify(vals=spec2)


"""Interface functions"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(handle_data)

spectrogram = browserinterface.add_handler(name="Heat Map", type='base:spectrogram',
        parameters={'width': width, 'height': height, 'min': min_val, 'max': max_val, 'delta_freq': sampling_freq / fft_size})

"""START"""
browserinterface.start()
browserinterface.change_config(channels=num_channels, buffer_frames=buffer_size, volume=80, rate=sampling_freq)
browserinterface.loop_callbacks()

