import numpy as np
import sys
sys.path.append('..')
import browserinterface

import realtimeaudio as rt
from math import ceil


""" Board parameters """
buffer_size = 4096
sampling_freq = 44100

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

f = 100

def when_config(buffer_frames, rate, channels, volume):
    global chart, fs, buffer_size, d, vad, f, nothing
    if channels != 2 or buffer_frames != 4096:
        browserinterface.change_config(channels=2, buffer_frames=4096)
        return
    chart = browserinterface.add_handler("Speech", 'base:graph:line', {'xName': 'Duration', 'min': -10000, 'max': 10000, 'xLimitNb': (rate/f*10), 'series': [{'name': 'Voice', 'color': 'green'}, {'name': 'Other', 'color': 'red'}]})
    fs = rate
    buffer_size = buffer_frames
    nothing = np.zeros(int(ceil(buffer_frames/float(f)))).tolist()
    d = rt.transforms.DFT(nfft=buffer_size)
    vad = rt.VAD(buffer_size, fs, 10, 40e-3, 3, 1.2)

i = 0
# perform VAD
def apply_vad(buffer):
    global chart, fs, buffer_size, d, vad, i, f, nothing, led_ring
    # grab slice and take DFT to feed to VAD
    sig = buffer[:,0]  # only one channel
    X = d.analysis(sig)

    # perform VAD
    decision = vad.decision(X)
    # decision = vad.decision_energy(sig, 4)
    t = np.linspace(i*buffer_size,(i+1)*buffer_size,buffer_size)/float(fs)
    # o1 = {'x': t[::f].tolist(), 'y': nothing}
    o1 = {'x': [], 'y': []}
    o2 = {'x': t[::f].tolist(), 'y': sig[::f].tolist()}
    if decision:
        chart.send_data({'add':[o2, o1]})
        if led_ring: led_ring.lightify_mono(rgb=[0,255,0],realtime=True)
    else:
        chart.send_data({'add':[o1, o2]})
        if led_ring: led_ring.lightify_mono(rgb=[255,0,0],realtime=True)
    i += 1


"""Interface features"""
browserinterface.register_when_new_config(when_config)
browserinterface.register_handle_data(apply_vad)

"""START"""
browserinterface.change_config(channels=2, buffer_frames=buffer_size,
    rate=sampling_freq, volume=80)
browserinterface.start()
browserinterface.loop_callbacks()
