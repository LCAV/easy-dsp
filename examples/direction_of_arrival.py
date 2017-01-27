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
sampling_freq = 48000

"""Select appropriate microphone array"""
# mic_array = rt.bbb_arrays.R_compactsix_random
mic_array = rt.bbb_arrays.R_compactsix_circular_1

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

"""Initialization block"""
def init(buffer_frames, rate, channels, volume):
    global doa, dft, vad, nfft, buffer_size, mic_array

    if channels != 6:
        browserinterface.change_config(channels=6, 
            buffer_frames=buffer_frames)

    # dft = rt.transforms.DFT(nfft=buffer_size)
    # vad = rt.VAD(buffer_size, rate, 10, 40e-3, 3, 1.2)

    # doa = rt.doa.SRP(mic_array, rate, nfft, n_grid=36)
    doa = rt.doa.MUSIC(mic_array, rate, nfft, n_grid=36)

"""Callback"""
def apply_doa(audio):
    global doa, dft, vad, nfft, buffer_size, led_ring

    if (browserinterface.buffer_frames != buffer_size 
        or browserinterface.channels != 6):
        print("Did not receive expected audio!")
        return

    # compute frequency domain snapshots
    hop_size = int(nfft/2)
    n_snapshots = int(np.floor(buffer_size/hop_size))-1
    X_stft = rt.utils.compute_snapshot_spec(audio, nfft, 
        n_snapshots, hop_size)

    # perform direction of arrival
    doa.locate_sources(X_stft)

    # send to browser for visualization
    to_send = doa.grid.values.tolist()
    to_send.append(to_send[0])
    polar_chart.send_data([{ 'replace': to_send }])

    # send to lights if available
    if led_ring:
        led_ring.lightify(vals=doa.grid.values, realtime=True)

"""Interface features"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(apply_doa)
polar_chart = browserinterface.add_handler("Directions", 'base:polar:line', {'title': 'Direction', 'series': ['Intensity'], 'rmin': 0, 'rmax': 1.25, 'legend': {'from': 0, 'to': 370, 'step': 10}})

browserinterface.change_config(channels=6, buffer_frames=buffer_size,
    rate=sampling_freq, volume=100)
browserinterface.start()
browserinterface.loop_callbacks()
