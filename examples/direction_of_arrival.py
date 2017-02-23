import sys
import numpy as np

sys.path.append('..')
import browserinterface
import realtimeaudio as rt


"""
Number of snapshots for DOA will be: ~2*buffer_size/nfft
"""
buffer_size = 4096
nfft = 512
num_angles = 60

"""Select appropriate microphone array"""
#mic_array = rt.bbb_arrays.R_compactsix_random
mic_array = rt.bbb_arrays.R_compactsix_circular_1
sampling_freq = 48000

"""
Select frequency range
"""
n_bands = 25
freq_range = [1000., 3500.]
f_min = int(np.round(freq_range[0]/sampling_freq*nfft))
f_max = int(np.round(freq_range[1]/sampling_freq*nfft))
range_bins = np.arange(f_min, f_max+1)

"""Check for LED Ring"""
try:
    from neopixels import NeoPixels
    import matplotlib.cm as cm
    led_ring = NeoPixels(usb_port='/dev/cu.usbmodem1421',
        colormap=cm.afmhot)
    print("LED ring ready to use!")
except:
    print("No LED ring available...")
    led_ring = False

"""Initialization block"""
def init(buffer_frames, rate, channels, volume):
    global doa

    doa_args = {
            'L': mic_array,
            'fs': rate,
            'nfft': nfft,
            'num_src': 1,
            'n_grid': num_angles
            }

    # doa = rt.doa.SRP(**doa_args)
    doa = rt.doa.MUSIC(**doa_args)
    # doa = rt.doa.FRIDA(max_four=2, signal_type='visibility', G_iter=1, **doa_args)

"""Callback"""
def apply_doa(audio):
    global doa, nfft, buffer_size, led_ring

    if (browserinterface.buffer_frames != buffer_size 
        or browserinterface.channels != 6):
        print("Did not receive expected audio!")
        return

    # compute frequency domain snapshots
    hop_size = int(nfft/2)
    n_snapshots = int(np.floor(buffer_size/hop_size))-1
    X_stft = rt.utils.compute_snapshot_spec(audio, nfft, 
        n_snapshots, hop_size)

    # pick bands with most energy and perform DOA
    bands_pwr = np.mean(np.sum(np.abs(X_stft[:,range_bins,:])**2, axis=0), axis=1)
    freq_bins = np.argsort(bands_pwr)[-n_bands:] + f_min
    doa.locate_sources(X_stft, freq_bins=freq_bins)
    #doa.locate_sources(X_stft, freq_range=freq_range)

    # send to browser for visualization
    #if doa.grid.values.max() > 1:
    #    doa.grid.values /= doa.grid.values.max()
    to_send = doa.grid.values.tolist()
    to_send.append(to_send[0])
    polar_chart.send_data([{ 'replace': to_send }])

    # send to lights if available
    if led_ring:
        led_ring.lightify(vals=doa.grid.values[::-1], realtime=True)

"""Interface features"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(apply_doa)

polar_chart = browserinterface.add_handler(name="Directions", 
    type='base:polar:line', 
    parameters={'title': 'Direction', 'series': ['Intensity'], 
    'numPoints': num_angles} )

"""START"""
browserinterface.change_config(channels=6, buffer_frames=buffer_size,
    rate=sampling_freq, volume=80)
browserinterface.start()
browserinterface.loop_callbacks()
