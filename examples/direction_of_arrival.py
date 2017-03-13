import numpy as np
import colorsys

import browserinterface
import algorithms as rt

"""
This script runs real-time direction of arrival
finding algorithms. Possible algorithms can be selected here:
"""

""" Select algorithm """
doa_algo = 'SRPPHAT'
#doa_algo = 'MUSIC'
doa_algo_config = dict(
        MUSIC=dict(vrange=[0.1, 0.8]),
        SRPPHAT=dict(vrange=[0.1, 0.4]),
        )

"""
Number of snapshots for DOA will be: ~2*buffer_size/nfft
"""
buffer_size = 1024
num_channels=6
nfft = 512
num_angles = 60
transform = 'mkl'

"""
Select frequency range
"""
n_bands = 20
freq_range = [1000., 3500.]
use_bin = True  # use top <n_bands> frequencies (True) or use all frequencies within specified range (False)

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


"""Check for LED Ring"""
try:
    import matplotlib.cm as cm
    led_ring = rt.neopixels.NeoPixels(
            usb_port=led_ring_address,
            colormap=cm.winter, 
            vrange=doa_algo_config[doa_algo]['vrange']
            )
    num_pixels = led_ring.num_pixels
    print("LED ring ready to use!")
except:
    print("No LED ring available...")
    num_pixels = 60
    led_ring = False

# Color stuff
sym_ind = np.concatenate((np.arange(0, 30), -np.arange(1,31)[::-1]))
P = np.zeros((num_pixels, 3), dtype=np.float)
background = np.array([1/6., 0.2, 0.1])
source = np.array([0., 0.9, 1.])

map_val = np.zeros(num_pixels)
ff = 0.6

def make_colors(powers):
    global old_azimuths, map_val

    P[:,:] = 0

    # background color
    for i in range(num_pixels):
        P[i,:] = background

    # forget!
    map_val *= ff

    # source colors
    for i in range(num_pixels):

        # adjust range of power
        vrange = doa_algo_config[doa_algo]['vrange']
        value = (powers[i] - vrange[0]) / (vrange[1] - vrange[0])

        # clamp the values
        if value > 1:
            value = 1

        if value < 0.0:
            value = 0.0

        map_val[i] += (1 - ff) * value
            

        P[i,:] = colorsys.hsv_to_rgb(*(map_val[i] * source + (1 - map_val[i]) * background))
        
    led_ring.send_colors(P[::-1,:])


"""Initialization block"""
print("Using " + doa_algo)
def init(buffer_frames, rate, channels, volume):
    global doa

    doa_args = {
            'L': mic_array,
            'fs': rate,
            'nfft': nfft,
            'num_src': 1,
            'n_grid': num_angles
            }

    # doa = rt.doa.FRIDA(max_four=2, signal_type='visibility', G_iter=1, **doa_args)
    if doa_algo == 'SRPPHAT':
        doa = rt.doa.SRP(**doa_args)
    elif doa_algo == 'MUSIC':
        doa = rt.doa.MUSIC(**doa_args)
    elif doa_algo == 'CSSM':
        doa = rt.doa.CSSM(num_iter=1, **doa_args)
    elif doa_algo == 'WAVES':
        doa = rt.doa.WAVES(num_iter=1, **doa_args)
    elif doa_algo == 'TOPS':
        doa = rt.doa.TOPS(**doa_args)


"""Callback"""
f_min = int(np.round(freq_range[0]/sampling_freq*nfft))
f_max = int(np.round(freq_range[1]/sampling_freq*nfft))
range_bins = np.arange(f_min, f_max+1)
def apply_doa(audio):
    global doa, nfft, buffer_size, led_ring

    # check for correct audio shape
    if audio.shape != (buffer_size, num_channels):
        print("Did not receive expected audio!")
        return

    # compute frequency domain snapshots
    hop_size = int(nfft/2)
    n_snapshots = int(np.floor(buffer_size/hop_size))-1
    X_stft = rt.utils.compute_snapshot_spec(audio, nfft, 
        n_snapshots, hop_size, transform=transform)

    # pick bands with most energy and perform DOA
    if use_bin:
        bands_pwr = np.mean(np.sum(np.abs(X_stft[:,range_bins,:])**2, axis=0), axis=1)
        freq_bins = np.argsort(bands_pwr)[-n_bands:] + f_min
        doa.locate_sources(X_stft, freq_bins=freq_bins)
    else:
        doa.locate_sources(X_stft, freq_range=freq_range)

    # send to browser for visualization
    to_send = doa.grid.values.tolist()
    to_send.append(to_send[0])
    polar_chart.send_data([{ 'replace': to_send }])

    # send to lights if available
    if led_ring:
        make_colors(doa.grid.values)
        #led_ring.lightify(vals=doa.grid.values, realtime=True)

"""Interface features"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(apply_doa)

polar_chart = browserinterface.add_handler(name="Directions", 
    type='base:polar:line', 
    parameters={'title': 'Direction', 'series': ['Intensity'], 
    'numPoints': num_angles} )

"""START"""
browserinterface.start()
browserinterface.change_config(channels=num_channels, buffer_frames=buffer_size,
    rate=sampling_freq, volume=80)
browserinterface.loop_callbacks()

