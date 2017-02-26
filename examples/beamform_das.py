import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
import browserinterface
import realtimeaudio as rt

"""Select appropriate microphone array"""
mic_array = rt.bbb_arrays.R_compactsix_random; sampling_freq = 44100
# mic_array = rt.bbb_arrays.R_compactsix_circular_1; sampling_freq = 48000

"""capture parameters"""
zero_padding = 100
buffer_size = 8192-zero_padding/2
num_channels = 6

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


"""Setup"""
num_angles = 60 # for directivity
direction = 70 # degrees
def init(buffer_frames, rate, channels, volume):
    global stft, bf

    stft = rt.transforms.STFT(2*buffer_size, rate, num_sig=num_channels)
    stft.zero_pad_back(zero_padding); stft.reset()

    bf = rt.beamformers.DAS(mic_array, sampling_freq, direction=direction, nfft=stft.nfft, num_angles=num_angles)
    stft.H = bf.weights.T

    # visualization
    freq_viz = 2000 # frequency for which to visualize beam pattern
    beam_shape = bf.get_directivity(freq=freq_viz)
    beam = beam_shape.tolist()
    beam.append(beam[0]) # "close" beam shape
    polar_chart.send_data([{ 'replace': beam }])
    if led_ring:
        led_ring.lightify(vals=beam_shape)


"""Defining callback"""
def beamform_audio(audio):
    global stft, bf

    stft.analysis(audio)
    stft.process()
    y = np.sum(stft.synthesis(), axis=1)

    # This should work to send back audio to browser
    audio[:,0] = y.astype(audio.dtype)
    audio[:,1] = y.astype(audio.dtype)
    audio[:,2:] = 0

    browserinterface.send_audio(audio)


"""Interface features"""
browserinterface.register_when_new_config(init)
browserinterface.register_handle_data(beamform_audio)
polar_chart = browserinterface.add_handler(name="Beam pattern", 
    type='base:polar:line', 
    parameters={'title': 'Beam pattern', 'series': ['Intensity'], 
    'numPoints': num_angles} )


"""START"""
browserinterface.change_config(buffer_frames=buffer_size, 
    channels=num_channels, rate=sampling_freq, volume=80)
browserinterface.start()
browserinterface.loop_callbacks()
