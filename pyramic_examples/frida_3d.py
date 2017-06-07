from __future__ import division, print_function
import numpy as np
import colorsys
import time

import browserinterface
import algorithms as rt

"""
Number of snapshots for DOA will be: ~2*buffer_size/nfft
"""
array_type = 'pyramic_full'
sampling_freq = 48000
buffer_size = 14400; num_channels = 48
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
    led_ring_address = config['led_ring_address']
except:
    # default when no hw config file is present
    led_ring_address = '/dev/cu.usbmodem1421'

"""Select appropriate microphone array"""
if array_type == 'pyramic_flat':
    mic_array, channel_mapping = rt.pyramic_tetrahedron.get_pyramic(dim=2)
elif array_type == 'pyramic_full':
    mic_array, channel_mapping = rt.pyramic_tetrahedron.get_pyramic(dim=3)
elif array_type == 'pyramic_subset':
    channel_mapping = list(range(48))[::2]
    mic_array, _ = rt.pyramic_tetrahedron.get_pyramic(dim=3)
    mic_array = mic_array[:, channel_mapping]

print(mic_array.shape)

doa_args = {
        'dim': mic_array.shape[0],
        'num_src': num_src,
        'n_grid': num_angles,
        'max_four': 4,
        'max_ini': 10,
        'max_iter': 2,
        'G_iter': 1,
        'low_rank_cleaning': False,
        'signal_type': 'visibility',
        'use_cache': False,
        }


"""
Select frequency range
"""
n_bands = 17
freq_range = [2000., 4000.]
f_min = int(np.round(freq_range[0] / sampling_freq*nfft))
f_max = int(np.round(freq_range[1] / sampling_freq*nfft))
range_bins = np.arange(f_min, f_max+1)
freq_bins = np.round(np.linspace(freq_range[0], freq_range[1], n_bands) / sampling_freq * nfft)

vrange = [-3., -0.5]

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
led_rot_offset = 17  # mismatch of led ring and microphone array

# a Bell curve for visualization
sym_ind = np.concatenate((np.arange(0, 30), -np.arange(1,31)[::-1]))
P = np.zeros((num_pixels, 3), dtype=np.float)
source_hue = [0., 0.4]
source_sat = [0.9, 0.8]
background = np.array(colorsys.hsv_to_rgb(0.45, 0.2, 0.1))
source = np.array(colorsys.hsv_to_rgb(0.11, 0.9, 1.))

col_interval = np.array([0, 0.6 * np.pi])
hue_leaky = source_hue[0]

map_val = np.zeros(num_pixels)
ff = 0.6

def make_colors(azimuth, colatitude, power):
    global map_val , hue_leaky

    P[:,:] = 0

    # background color
    for i in range(num_pixels):
        P[i,:] = background

    # forget!
    map_val *= ff

    # compute bin location, led array is in the other direction
    az_i = num_pixels - 1 - int(round(num_pixels * azimuth / (2 * np.pi))) % num_pixels

    hue = np.minimum(col_interval[1], colatitude) / col_interval[1] * (source_hue[1] - source_hue[0]) + source_hue[0]
    hue_leaky = 0.5 * hue_leaky + 0.5 * hue

    # adjust range of power
    value = (np.log10(power) - vrange[0]) / (vrange[1] - vrange[0])

    # clamp the values
    if value > 1:
        value = 1

    if value < 0.0:
        value = 0.0

    # set the direction
    if (value > 0.5):
        map_val[az_i] = value
        map_val[(az_i-1)%num_pixels] = 0.6 * value
        map_val[(az_i+1)%num_pixels] = 0.6 * value

    else:
        map_val[az_i] += (1 - ff) * value
        map_val[(az_i-1)%num_pixels] += (1 - ff) * value * 0.6
        map_val[(az_i+1)%num_pixels] += (1 - ff) * value * 0.6

    source_col = np.array(colorsys.hsv_to_rgb(hue_leaky, 0.9, 1.))

    for i in range(num_pixels):
        P[i,:] = map_val[i] * source_col + (1 - map_val[i]) * background
        #P[i,:] = map_val[i] * source_col

    led_ring.send_colors(np.roll(P, led_rot_offset, axis=0))

"""Initialization block"""
def init(buffer_frames, rate, channels, volume):
    global doa, doa_args, sampling_freq, nfft, mic_array

"""Callback"""
def apply_doa(audio):
    global doa, nfft, buffer_size, led_ring, channel_mapping

    # check for correct audio shape
    if audio.shape != (buffer_size, num_channels):
        print("Did not receive expected audio!")
        return

    # compute frequency domain snapshots
    tic = time.time()
    hop_size = nfft // 2
    n_snapshots = int(np.floor(buffer_size / hop_size))-1
    X_stft = rt.utils.compute_snapshot_spec(audio[:,channel_mapping], nfft, 
        n_snapshots, hop_size, transform=transform)
    stft_time = time.time() - tic

    # pick bands with most energy and perform DOA
    tic = time.time()
    doa.locate_sources(X_stft, freq_bins=freq_bins)
    doa_time = time.time() - tic

    # send to browser for visualization
    # Now map the angles to some function
    # send to lights if available
    pwr = np.abs(doa.alpha_recon).max(axis=1)
    line = '{:6.3f} az={:8.2f} co={:8.2f} stft_time={:.3f} doa_time={:.3f}'.format(np.log10(pwr[0]), 
            doa.azimuth_recon[0] / np.pi * 180, doa.colatitude_recon[0] / np.pi * 180, stft_time, doa_time)
    print(line)
    if led_ring:
        make_colors(doa.azimuth_recon[0], doa.colatitude_recon[0], np.max(np.abs(doa.alpha_recon[0])))




# run once to initialize everything
doa = rt.doa.FRIDA(mic_array, sampling_freq, nfft, **doa_args)
apply_doa((np.random.randn(buffer_size, num_channels) * 100).astype(np.int16))


"""Interface features"""
browserinterface.inform_browser = False
browserinterface.bi_board_ip = '192.168.2.26'
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(apply_doa)

"""START"""
browserinterface.start()
browserinterface.change_config(channels=num_channels, 
    buffer_frames=buffer_size, rate=sampling_freq, volume=80)
browserinterface.loop_callbacks()
