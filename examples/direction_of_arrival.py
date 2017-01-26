import sys
import numpy as np

sys.path.append('..')
import browserinterface

import realtimeaudio as rt

doa = 0
nfft = 256
xvals = np.linspace(0, 2*np.pi, 60, endpoint=False)

# mic_array = rt.bbb_arrays.R_compactsix_random
mic_array = rt.bbb_arrays.R_compactsix_circular_1

def init(buffer_frames, rate, channels, volume):
    global doa, nfft, mic_array, d, vad
    if channels != 6 or buffer_frames < 4096:
        browserinterface.change_config(channels=6, buffer_frames=4096)

    d = rt.transforms.DFT(nfft=buffer_frames)
    vad = rt.VAD(buffer_frames, rate, 10, 40e-3, 3, 1.2)

    # create DOA object
    doa = rt.doa.SRP(mic_array, rate, nfft, n_grid=36)
    # doa = rt.doa.MUSIC(mic_array, rate, nfft, n_grid=36)

browserinterface.register_when_new_config(init)

polar_chart = browserinterface.add_handler("Directions", 'base:polar:line', {'title': 'Direction', 'series': ['Intensity'], 'rmin': 0, 'rmax': 1.25, 'legend': {'from': 0, 'to': 370, 'step': 10}})
no_speech = np.zeros(36)

def handle_data(buffer):
    global doa, nfft, d, vad

    if browserinterface.buffer_frames < 4096 or browserinterface.channels != 6:
        return

    # print "Buffer received"
    # parameters
    buffer_size = browserinterface.buffer_frames
    hop_size = int(nfft/2)
    n_snapshots = int(np.floor(buffer_size/hop_size))-1

    x = buffer
    fs = browserinterface.rate

    # perform VAD
    X = d.analysis(x[:,0])
    decision = vad.decision(X)
    if decision:
        # perform DOA
        X_stft = rt.utils.compute_snapshot_spec(x, nfft, n_snapshots, hop_size)
        doa.locate_sources(X_stft)
        to_send = doa.grid.values.tolist()
        to_send.append(to_send[0])
        polar_chart.send_data([{ 'replace': to_send }])
    else:
        to_send = no_speech.tolist()
        to_send.append(to_send[0])
        polar_chart.send_data([{ 'replace': to_send }])
        

browserinterface.register_handle_data(handle_data)
browserinterface.start()
browserinterface.loop_callbacks()
