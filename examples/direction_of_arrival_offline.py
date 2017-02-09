import sys
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import realtimeaudio as rt


"""
Number of snapshots for DOA will be: ~2*buffer_size/nfft
"""
buffer_size = 4096
nfft = 512
num_angles = 360

"""Select appropriate microphone array"""
mic_array = rt.bbb_arrays.R_compactsix_random
# mic_array = rt.bbb_arrays.R_compactsix_circular_1

"""Check for LED Ring"""
try:
    from neopixels import NeoPixels
    import matplotlib.cm as cm
    led_ring = NeoPixels(usb_port='/dev/cu.usbmodem1411',
        colormap=cm.afmhot)
    print("LED ring ready to use!")
except:
    print("No LED ring available...")
    led_ring = False

# read recording
sampling_freq, x = wavfile.read('out.wav')

# create DOA object
doa_args = {
        'L': mic_array,
        'fs': sampling_freq,
        'nfft': nfft,
        'num_src': 1,
        'n_grid': num_angles
        }
# doa = rt.doa.SRP(**doa_args)
# doa = rt.doa.MUSIC(**doa_args)
doa = rt.doa.FRIDA(max_four=5, signal_type='visibility', G_iter=1, **doa_args)

# choose frequency range
n_bands = 5
freq_range = [100., 4500.]
f_min = int(np.round(freq_range[0]/sampling_freq*nfft))
f_max = int(np.round(freq_range[1]/sampling_freq*nfft))
range_bins = np.arange(f_min, f_max+1)

# perform DOA
n_frames = 10
hop_size = int(nfft/2)
n_snapshots = int(np.floor(buffer_size/hop_size))-1
for i in range(n_frames):

    signals = x[buffer_size*i:buffer_size*(i+1),:]
    X_stft = rt.utils.compute_snapshot_spec(signals, nfft, n_snapshots, hop_size)

    bands_pwr = np.mean(np.sum(np.abs(X_stft[:,range_bins,:])**2, axis=0), axis=1)
    freq_bins = np.argsort(bands_pwr)[-n_bands:] + f_min

    doa.locate_sources(X_stft, freq_bins=freq_bins)
    doa.polar_plt_dirac()
    plt.show()
