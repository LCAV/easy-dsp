import numpy as np
import colorsys

import browserinterface
import algorithms as rt

"""
Number of snapshots for DOA will be: ~2*buffer_size/nfft
"""
buffer_size = 8192; num_channels=6
nfft = 512
num_angles = 60
num_src = 1
transform = 'mkl'

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

"""
Select frequency range
"""
n_bands = 20
freq_range = [1000., 3500.]
f_min = int(np.round(freq_range[0]/sampling_freq*nfft))
f_max = int(np.round(freq_range[1]/sampling_freq*nfft))
range_bins = np.arange(f_min, f_max+1)
use_bin = False

vrange = [0.1, 5.]

"""Check for LED Ring"""
try:
    import matplotlib.cm as cm
    led_ring = rt.neopixels.NeoPixels(usb_port=led_ring_address,
        colormap=cm.afmhot, vrange=vrange)
    print("LED ring ready to use!")
    num_pixels = led_ring.num_pixels
except:
    print("No LED ring available...")
    led_ring = False
    num_pixels = 60

# a Bell curve for visualization
sym_ind = np.concatenate((np.arange(0, 30), -np.arange(1,31)[::-1]))
P = np.zeros((num_pixels, 3), dtype=np.float)
old_azimuths = np.zeros(num_src)
source_hue = [0.11, 0.5]
source_sat = [0.9, 0.8]
background = np.array(colorsys.hsv_to_rgb(0.45, 0.2, 0.1))
source = np.array(colorsys.hsv_to_rgb(0.11, 0.9, 1.))
spatial_spectrum = np.zeros(num_pixels)

map_val = np.zeros(num_pixels)
ff = 0.7

def make_colors(azimuths, powers):
    global old_azimuths, map_val

    P[:,:] = 0
    spatial_spectrum[:] = 0.1

    # background color
    for i in range(num_pixels):
        P[i,:] = background

    # forget!
    map_val *= ff

    # source colors
    for azimuth, power, hue, sat in zip(azimuths, powers, source_hue, source_sat):

        # compute bin location, led array is in the other direction
        i = num_pixels - 1 - int(round(num_pixels * azimuth / (2 * np.pi))) % num_pixels

        # adjust range of power
        value = (np.log10(power) - vrange[0]) / (vrange[1] - vrange[0])

        # clamp the values
        if value > 1:
            value = 1

        if value < 0.0:
            value = 0.0

        # set the direction
        if (value > 0.5):
            map_val[i] = value
            map_val[(i-1)%num_pixels] = 0.6 * value
            map_val[(i+1)%num_pixels] = 0.6 * value

        else:
            map_val[i] += (1 - ff) * value
            map_val[(i-1)%num_pixels] += (1 - ff) * value * 0.6
            map_val[(i+1)%num_pixels] += (1 - ff) * value * 0.6

        spatial_spectrum[i] += value

    for i in range(num_pixels):
        P[i,:] = map_val[i] * source + (1 - map_val[i]) * background

    led_ring.send_colors(P)


"""Initialization block"""
def init(buffer_frames, rate, channels, volume):
    global doa

    doa_args = {
            'L': mic_array,
            'fs': rate,
            'nfft': nfft,
            'num_src': num_src,
            'n_grid': num_angles,
            'max_four': 4,
            'max_ini': 10,
            'max_iter': 3,
            'G_iter': 1,
            'low_rank_cleaning': True,
            'signal_type': 'visibility',
            }

    doa = rt.doa.FRIDA(**doa_args)

"""Callback"""
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
    # Now map the angles to some function
    # send to lights if available
    if led_ring:
        make_colors(doa.azimuth_recon, doa.alpha_recon.mean(axis=1))

    to_send = spatial_spectrum.tolist()
    to_send.append(to_send[0])
    polar_chart.send_data([{ 'replace': to_send }])


"""Interface features"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(apply_doa)

polar_chart = browserinterface.add_handler(name="Directions", 
    type='base:polar:line', 
    parameters={'title': 'Direction', 'series': ['Intensity'], 
    'numPoints': num_angles} )

"""START"""
browserinterface.start()
browserinterface.change_config(channels=num_channels, 
    buffer_frames=buffer_size, rate=sampling_freq, volume=80)
browserinterface.loop_callbacks()
