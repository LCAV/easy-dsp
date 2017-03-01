import sys
import numpy as np
import matplotlib.pyplot as plt

import browserinterface
import algorithms as rt

"""
Read hardware config from file
"""
try:
    import json
    with open('./hardware_config.json', 'r') as config_file:
        config = json.load(config_file)
        config_file.close()
    sampling_freq = config['sampling_frequency']
    array_type = config['array_type']
    led_ring_address = config['led_ring_address']
except:
    # default when no hw config file is present
    sampling_freq = 44100
    array_type = 'random'
    led_ring_address = '/dev/cu.usbmodem1421'

"""Select appropriate microphone array"""
if array_type == 'random':
    mic_array = rt.bbb_arrays.R_compactsix_random
elif array_type == 'circular':
    mic_array = rt.bbb_arrays.R_compactsix_circular_1

"""define capture parameters accordingly"""
zero_padding = 100
nfft = 16384
buffer_size = nfft/2-zero_padding/2
num_channels = 6

"""Check for LED Ring"""
try:
    import matplotlib.cm as cm
    led_ring = rt.neopixels.NeoPixels(usb_port=led_ring_address,
        colormap=cm.afmhot)
    print("LED ring ready to use!")
except:
    print("No LED ring available...")
    led_ring = False


"""Setup"""
num_angles = 60 # for directivity
direction = 70 # degrees
def init(buffer_frames, rate, channels, volume):
    global stft, bf

    stft = rt.transforms.STFT(2*buffer_size, rate, num_sig=num_channels)

    bf = rt.beamformers.DAS(mic_array, sampling_freq, direction=direction, nfft=nfft, num_angles=num_angles)
    stft.set_filter(coeff=bf.weights, freq=True, zb=zero_padding)

    # visualization
    freq_viz = 2000 # frequency for which to visualize beam pattern
    beam_shape = bf.get_directivity(freq=freq_viz)
    beam = beam_shape.tolist()
    beam.append(beam[0]) # "close" beam shape
    polar_chart.send_data([{ 'replace': beam }])
    if led_ring:
        led_ring.lightify(vals=beam_shape)


"""Defining callback"""
def beamform_audio(audio):
    global stft, bf

    if (audio.shape[0] != browserinterface.buffer_frames 
        or audio.shape[1] != browserinterface.channels):
        print("Did not receive expected audio!")
        return

    stft.analysis(audio)
    stft.process()
    y = np.sum(stft.synthesis(), axis=1)

    # This should work to send back audio to browser
    audio[:,0] = y.astype(audio.dtype)
    audio[:,1] = y.astype(audio.dtype)
    audio[:,2:] = 0

    browserinterface.send_audio(audio)


"""Interface features"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(beamform_audio)
polar_chart = browserinterface.add_handler(name="Beam pattern", 
    type='base:polar:line', 
    parameters={'title': 'Beam pattern', 'series': ['Intensity'], 
    'numPoints': num_angles} )


"""START"""
browserinterface.start()
browserinterface.change_config(buffer_frames=buffer_size, 
    channels=num_channels, rate=sampling_freq, volume=80)
browserinterface.loop_callbacks()
