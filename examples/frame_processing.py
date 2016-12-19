"""
Apply processing frame by frame in the STFT domain (overlapping windows).
"""
import sys
sys.path.append('..')
sys.path.append('examples')

import browserinterface
import numpy as np
from stft import STFT
import windows

num_windows = 2
s = 0

def init(buffer_frames, rate, channels, volume):
    global s, num_windows, N, hop
    if channels != 2 or buffer_frames != 8192:
        browserinterface.change_config(channels=2, buffer_frames=8192)
        return

    # parameters (block size, number of signals, hop size)
    N = browserinterface.buffer_frames/num_windows
    hop = int(np.floor(N/2))

    # filter (moving average --> low pass) and necessary zero padding
    filter_length = 50
    h = windows.rect(filter_length)/(filter_length*1.0)
    zf = filter_length/2
    zb = filter_length/2

    # create STFT object
    s = STFT(N,browserinterface.rate,hop,zf=zf,zb=zb,h=h)

browserinterface.register_when_new_config(init)

c_magnitude = browserinterface.add_handler("Magnitude", 'base:graph:line', {'min': 0, 'max': 100000, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})
c_magnitude_f = browserinterface.add_handler("Magnitude Filtered", 'base:graph:line', {'min': 0, 'max': 100000, 'xName': 'Frequency', 'series': [{'name': '1'}, {'name': '2'}]})

def handle_data(buffer):
    global s, num_windows, N, hop

    if browserinterface.channels != 2 or browserinterface.buffer_frames != 8192:
        return

    x = buffer
    fs = browserinterface.rate

    num_sig = x.shape[1]

    # apply stft and istft for a few windows
    for i in range(num_windows):
        # read frame --> would read <hop> number of samples here
        signals = x[hop*i:hop*(i+1),:]
        # perform analysis and visualize the frame
        s.analysis(signals[:,0])
        s.visualize_frame(c_magnitude, fmin=0.0,fmax=10000.0,plot_time=False)
        # apply filtering and visualize results
        s.process()
        s.visualize_frame(c_magnitude_f, fmin=0.0,fmax=10000.0,plot_time=False)
        # plt.show()
        # apply ISFTF to get filtered output in time domain
        xr = s.synthesis()

browserinterface.register_handle_data(handle_data)
browserinterface.start()
browserinterface.loop_callbacks()
