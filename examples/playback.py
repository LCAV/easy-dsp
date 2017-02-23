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
buffer_size = 4096
num_channels = 2

"""Defining callback"""
def example_callback(audio):

    # play back audio
    browserinterface.send_audio(audio)

browserinterface.register_handle_data(example_callback)


"""START"""
browserinterface.change_config(buffer_frames=buffer_size, 
    channels=num_channels, rate=sampling_freq, volume=80)
browserinterface.start()
browserinterface.loop_callbacks()
