
# First you import the module
import argparse
from scipy.io import wavfile

import browserinterface

import argparse

parser = argparse.ArgumentParser(description='Record from an easy-dsp enabled microphone array')
parser.add_argument('ip', type=str,
                    help='ip address of the microphone array')
parser.add_argument('filename', type=str, nargs='?', default='record.wav',
                    help='filename to save the recorded data')
parser.add_argument('-r', '--fs', type=int, default=48000,
                    help='sampling frequency')
parser.add_argument('-c', '--channels', type=int, default=2,
                    help='number of channels')
parser.add_argument('-b', '--buffer', type=int, default=1024,
                    help='buffer size')
parser.add_argument('-v', '--volume', type=int, default=90,
                    help='microphones volume')
parser.add_argument('-d', '--duration', type=int, default=5000,
                    help='record duration')

args = parser.parse_args()
print(args)

print('Record {:.2f}s @ fs={} ch={} buf_size={} vol={} from {} to {}'.format(
    args.duration / 1000, args.fs, args.channels, args.buffer, args.volume, args.ip, args.filename))

browserinterface.inform_browser = False
browserinterface.bi_board_ip = args.ip

is_recording = True

def save_audio(buffer):
    global is_recording
    wavfile.write(args.filename, args.fs, buffer)
    print("Audio has been recorded", len(buffer))
    is_recording = False

browserinterface.start()
browserinterface.change_config(rate=args.fs, channels=args.channels, buffer_frames=args.buffer, volume=args.volume)

browserinterface.record_audio(args.duration, save_audio) # my_function will be called after 15 seconds

while is_recording:
    browserinterface.process_callbacks()
