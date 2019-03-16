"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).

This is the callback (non-blocking) version.
"""

import pyaudio
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
from move_utils import *
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "utils"))
from debugger import Debugger
from data_writer_util import LabelFile


############### Set Up Stuff

WIDTH = 2
CHANNELS = 2
RATE = 44100

frames = []

def callback(in_data, frame_count, time_info, status):
    global frames
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)

def save_wav(data, name):
    global CHANNELS
    global WIDTH
    global RATE

    wf = wave.open(name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                stream_callback=callback)


#Set up

Walker = RandomPolarWalker()
Viz = Debugger()
#MAKE SURE TO CHANGE THE FILE NUMBER
File = LabelFile(0)

last_time = time.time()

##############################
"""IMPORTANT"""
rec_time = 1 * 60
label_freq = 100
##############################

#TODO Make it so you can get a chunck to pass to the NN. Then you come back and get the next best chunck

#i think its a case of closing it and then repoening.

time_running = 0
label_time = 0
label_period = 1/label_freq



# Wait for person to get into correct position
theta = Walker.heading()

count_down = 5
start_up_timer = 0
while start_up_timer < count_down:
    start_up_timer = time.time() - last_time
    Viz.draw_heading(theta, show = False)
    Viz.write(str(round(count_down - start_up_timer)))
    plt.show()
    plt.pause(0.000001)
#Record and Label

#Start and add first label at t = 0

stream.start_stream()
File.write_heading(theta)

last_time = time.time() # or else you do a massive dt to start
while time_running < rec_time:
    curr_time = time.time()
    dt = curr_time - last_time
    last_time = curr_time
    time_running += dt
    label_time += dt

    Walker.slow_update(dt)
    theta = Walker.heading()
    Viz.draw_heading(theta)

    if label_time >= label_period:
        label_time = 0
        File.write_heading(theta)

stream.stop_stream()
stream.close()

path = "/Users/zachyamaoka/Dropbox/de3_audio_data/data_wav_5/"
file_name = "real_rec_001.wav"
name=path+file_name

save_wav(frames, name)

p.terminate()
