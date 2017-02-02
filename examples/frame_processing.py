"""
Apply processing frame by frame in the STFT domain (overlapping windows).
"""
import sys
import numpy as np

sys.path.append('..')
import browserinterface

import realtimeaudio as rt


buffer_size = 8192
num_windows = 3
sampling_freq = 44100
# sampling_freq = 48000

def init(buffer_frames, rate, channels, volume):
    global stft

    # parameters (block size, number of signals, hop size)
    num_samples = browserinterface.buffer_frames/num_windows
    hop = int(np.floor(num_samples/2))

    # filter (moving average --> low pass) and necessary zero padding
    filter_length = 50
    h = rt.windows.rect(filter_length)/(filter_length*1.0)
    zf = filter_length/2
    zb = filter_length/2

    # create STFT object
    stft = rt.transforms.STFT(num_samples, rate, hop,zf=zf,zb=zb,h=h)

def visualize_spectrum(handle):
    global stft

    freq = np.linspace(0,sampling_freq/2,len(stft.X))
    freq = np.ceil(freq).tolist()[::5]
    sig = []

    sig.append({'x': freq, 'y': np.floor(abs(stft.X)).tolist()[::5]})
    handle.send_data({'replace': sig})

def handle_data(buffer):
    global stft

    if (browserinterface.buffer_frames != buffer_size 
        or browserinterface.channels != 2):
        print("Did not receive expected audio!")
        return

    # apply stft and istft for a few windows
    for i in range(num_windows):

        # perform analysis and visualize the frame
        stft.analysis(buffer[stft.hop*i:stft.hop*(i+1),0])
        visualize_spectrum(c_magnitude)

        # apply filtering and visualize results
        stft.process()
        visualize_spectrum(c_magnitude_f)

"""Interface functions"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(handle_data)
c_magnitude = browserinterface.add_handler("Magnitude", 'base:graph:line', {'min': 0, 'max': 100000, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})
c_magnitude_f = browserinterface.add_handler("Magnitude Filtered", 'base:graph:line', {'min': 0, 'max': 100000, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})


"""START"""
browserinterface.change_config(channels=2, buffer_frames=buffer_size, 
    volume=80, rate=sampling_freq)
browserinterface.start()
browserinterface.loop_callbacks()
