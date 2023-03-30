#!/usr/bin/env python3
#Example code from https://python-sounddevice.readthedocs.io/en/0.4.6/examples.html#plot-microphone-signal-s-in-real-time
import argparse
import queue
import sys
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal

#Use the below bool value to switch working modes
#Set both use_complex and use_hori to True for Part 4 of the lab
use_sin=False 
use_square=False
use_complex=True
use_hori=True
use_gib=False
######

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=15, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audios device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
parser.add_argument(
    '-a', '--amplitude', type=float, default=0.2,
    help='amplitude (default: %(default)s)')
parser.add_argument(
    'frequency', nargs='?', metavar='FREQUENCY', type=float, default=5000,
    help='frequency in Hz (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
print("mapping",mapping)
q = queue.Queue()
q2 = queue.Queue()
start_idx = 0
samplerate = sd.query_devices(args.device, 'output')['default_samplerate']
sampling_rate=44100

def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100,scale:float = 1) :
    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit * adata.size / sampling_rate)
    fsig = np.fft.fft(adata)
    
    fsig[bandlimit_index+1 : -bandlimit_index] = 0

    adata_filtered = np.fft.ifft(fsig)*scale
    real_data=np.real(adata_filtered)
    imag_data=np.imag(adata_filtered)


    return real_data,imag_data

def moving_average_lowpass_filter(signal, window_size):
    weights = np.ones(window_size) / window_size
    signal =np.squeeze(signal)
    filtered_signal = np.convolve(signal, weights, mode='valid')
    filtered_signal=np.expand_dims(filtered_signal, axis=1)
    return filtered_signal

def audio_callback(indata, outdata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    global start_idx    
    t = (start_idx + np.arange(frames)) / samplerate
    t = t.reshape(-1, 1)
    #sine wave
    if(use_sin):
        outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)
    if(use_square):
        outdata[:]=signal.square(2 * np.pi * 5000 * t, duty=0.5)
    if(use_complex):
        outdata[:] = args.amplitude * np.sin(2 * np.pi * args.frequency * t)
        phase_shift = args.amplitude * np.cos(2 * np.pi * args.frequency * t)
    if(use_gib):
        outdata[:]=5*(args.amplitude * np.sin(2 * np.pi * args.frequency * t)+ 0.3333*args.amplitude * np.sin(3*2 * np.pi * args.frequency * t) +\
                    0.2*args.amplitude * np.sin(5*2 * np.pi * args.frequency * t) + (1/7)*args.amplitude * np.sin(7*2 * np.pi * args.frequency * t) + \
                    (1/9)*args.amplitude * np.sin(9*2 * np.pi * args.frequency * t) + (1/11)*args.amplitude * np.sin(11*2 * np.pi * args.frequency * t))
   

    start_idx += frames
    # Fancy indexing with mapping creates a (necessary!) copy:
    mixer = indata*outdata
    if (use_sin):
        mixer_after_lowpass=moving_average_lowpass_filter(mixer,100)*50
        q.put(mixer_after_lowpass[::args.downsample, mapping])
    if (use_square):
        mixer_after_lowpass=moving_average_lowpass_filter(mixer,50)*10
        q.put(mixer_after_lowpass[::args.downsample, mapping])
    if (use_gib):
        q.put(outdata[::args.downsample, mapping])


    #real_lowpass,imag_lowpass=low_pass_filter(mixer,10,sampling_rate,10)
    #mixer_after_lowpass=moving_average_lowpass_filter(mixer,100)*50
    #avg_lowpass=np.average(mixer_after_lowpass,axis=1,keepdims=True))
    #q.put(indata[::args.downsample, mapping])
    if(use_complex):
        mixer_imag = phase_shift*indata
        real_after_lowpass=moving_average_lowpass_filter(mixer,100)*50
        imag_after_lowpass=moving_average_lowpass_filter(mixer_imag,100)*50
        q.put(real_after_lowpass[::args.downsample, mapping])
        q2.put(imag_after_lowpass[::args.downsample, mapping])


def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)

        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
        dot.set_ydata(np.average(data))


        if use_complex and not use_hori:
            try:
                complex_data = q2.get_nowait()
            except queue.Empty:
                break

            #if use_hori:
            #    dot2.set_xdata(np.average(complex_data))
            #else:
            #    dot2.set_ydata(np.average(complex_data))
            dot2.set_ydata(np.average(complex_data))

        if use_complex and use_hori:
            try:
                complex_data = q2.get_nowait()
            except queue.Empty:
                break
            dot.set_xdata(np.average(complex_data))
            #if use_hori:
            #    dot2.set_xdata(np.average(complex_data))
            #else:
            #    dot2.set_ydata(np.average(complex_data))

    if use_complex and not use_hori:
        return dot,dot2,
    else:
        return dot,


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = int(args.window * args.samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, len(args.channels)))

    fig, ax = plt.subplots()
    #lines = ax.plot(plotdata)
    dot, = ax.plot(0, 0, 'o', color="g",markersize=10)
    if use_complex and not use_hori:
        dot2, = ax.plot(0, 0, 'o', color="r",markersize=10)


    if len(args.channels) > 1:
        ax.legend([f'channel {c}' for c in args.channels],
                  loc='lower left', ncol=len(args.channels))
    print(len(plotdata))
    ax.axis((-1, 1, -1, 1))
    ax.set_yticks([0])
    #ax.yaxis.grid(True)
    ax.set_facecolor((0, 0, 0))

    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.Stream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    ani = FuncAnimation(fig, update_plot,interval=args.interval, blit=True)

    with stream:
        plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))