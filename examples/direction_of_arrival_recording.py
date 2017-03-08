import numpy as np

import browserinterface
import algorithms as rt
import time

""" Select algorithm """
doa_algo = 'SRPPHAT'
doa_algo = 'MUSIC'
doa_algo = 'TOPS'
doa_algo = 'CSSM'
doa_algo = 'WAVES'
doa_algo_config = dict(
        MUSIC=dict(vrange=[0.2, 0.6]),
        SRPPHAT=dict(vrange=[0.1, 0.4]),
        TOPS=dict(vrange=[0., 1.]),
        CSSM=dict(vrange=[0., 1.]),
        WAVES=dict(vrange=[0., 1.])
        )

"""
Select frequency range
"""
n_bands = 20
freq_range = [200., 4000.]
use_bin = False  # use top <n_bands> frequencies (True) or use all frequencies within specified range (False)
transform = 'mkl'

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
Number of snapshots for DOA will be: ~2*buffer_size/nfft
"""
buffer_size = 1024; num_channels=6
nfft = 512
num_angles = 60
transform = 'fftw'
num_angles = 360    # more angles typically yields better spatial resolution
n_frames = 10       # how many times to apply DOA
hop_size = nfft//2
n_snapshots = int(np.floor(buffer_size/hop_size))-1
recording_duration = float(buffer_size*n_frames)/sampling_freq   # in sec


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

"""
Create DOA object
"""
doa_args = {
        'L': mic_array,
        'fs': sampling_freq,
        'nfft': nfft,
        'num_src': 1,
        'n_grid': num_angles
        }

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


# apply DOA
f_min = int(np.round(freq_range[0]/sampling_freq*nfft))
f_max = int(np.round(freq_range[1]/sampling_freq*nfft))
range_bins = np.arange(f_min, f_max+1)
print("Using " + doa_algo)
def apply_doa(audio):
    global doa


    for i in range(n_frames):

        signals = audio[buffer_size*i:buffer_size*(i+1),:]
        X_stft = rt.utils.compute_snapshot_spec(signals, nfft, n_snapshots, hop_size)

        if use_bin:
            bands_pwr = np.mean(np.sum(np.abs(X_stft[:,range_bins,:])**2, axis=0), axis=1)
            freq_bins = np.argsort(bands_pwr)[-n_bands:] + f_min
            freq_hz = np.array(freq_bins, dtype=np.float32)/nfft*sampling_freq
            # print('Selected frequency bins: {0}'.format(freq_bins))
            # print('Selected frequencies: {0} Hertz'.format(freq_hz))
            doa.locate_sources(X_stft, freq_bins=freq_bins)
        else:
            # print('Frequency range: {0}'.format(freq_range))
            doa.locate_sources(X_stft, freq_range=freq_range)

        # send to browser for visualization
        if doa.grid.values.max() > 1:
            doa.grid.values /= doa.grid.values.max()
        to_send = doa.grid.values.tolist()
        to_send.append(to_send[0])
        polar_chart.send_data([{ 'replace': to_send }])

        # send to lights if available
        if led_ring:
            led_ring.lightify(vals=doa.grid.values, realtime=True)

        # time.sleep(float(buffer_size)/sampling_freq)
        time.sleep(1)
    

browserinterface.record_audio(recording_duration*1000, apply_doa)
polar_chart = browserinterface.add_handler(name="Directions", 
    type='base:polar:line', 
    parameters={'title': 'Direction', 'series': ['Intensity'], 
    'numPoints': num_angles} )

"""START"""
browserinterface.start()
browserinterface.change_config(buffer_frames=buffer_size, channels=num_channels, rate=sampling_freq, volume=80)
browserinterface.loop_callbacks()

