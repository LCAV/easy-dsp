from __future__ import division, print_function
import numpy as np

import browserinterface
import algorithms as rt
import time

"""Beamforming parameters"""
direction = 0 # degrees
transform = 'mkl' # 'numpy', 'mlk', 'fftw'

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
nfft = 2048  # seems that less than this is not reliable
buffer_size = nfft - zero_padding
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
num_angles = 60 # for directivity resolution
beam_shape = np.zeros(num_angles)
def init(buffer_frames, rate, channels, volume):
    global stft, bf

    stft = rt.transforms.STFT(buffer_size, rate, hop=buffer_size, channels=channels, transform=transform)

    # 
    bf = rt.beamformers.DAS(mic_array, sampling_freq, direction=direction, nfft=nfft, num_angles=num_angles)
    stft.set_filter(coeff=bf.weights, freq=True, zb=zero_padding)

    # visualization
    freq_viz = 2000 # frequency for which to visualize beam pattern
    beam_shape[:] = bf.get_directivity(freq=freq_viz)
    beam = beam_shape.tolist()
    beam.append(beam[0]) # "close" beam shape
    polar_chart.send_data([{ 'replace': beam }])

    if led_ring:
        led_ring.lightify(vals=beam_shape)

    # run once the processing on empty buffer to load all libraries
    stft.analysis(np.zeros((buffer_size, num_channels)))
    stft.process()
    stft.synthesis()


"""Defining callback"""
def beamform_audio(audio):
    global stft, bf

    # check for correct audio shape
    if audio.shape != (buffer_size, num_channels):
        print("Did not receive expected audio!")
        return

    # record start of processing time
    start_proc = time.time()

    # Do the filtering in frequency domain
    stft.analysis(audio)
    stft.process()

    # send back audio to browser
    audio[:,0] = np.sum(stft.synthesis(), axis=1)
    audio[:,1] = audio[:,0]
    audio[:,2:] = 0

    # Check time spent on processing
    proc_time = time.time() - start_proc
    if proc_time > buffer_size / sampling_freq:
        print("Processing is a little long:", proc_time, "sec vs", 
                buffer_size / sampling_freq, " sec available")

    if led_ring:
        led_ring.lightify(vals=beam_shape)

    # Send audio back to the browser
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
