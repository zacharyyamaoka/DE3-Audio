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
from data_utils import get_zero_string
from AudioLive import LivePlayer



WINDOW_TIME = 0.25

Player = LivePlayer(window = WINDOW_TIME, playback=True)

Walker = RandomPolarWalker()
Viz = Debugger()

#MAKE SURE TO CHANGE THE FILE NUMBER

File = LabelFile(num=DATA_N,stem="real_rec_",path="/Users/zachyamaoka/Documents/de3_audio/data_real_label/")

last_time = time.time()

##############################
"""IMPORTANT"""
rec_time = 10 * 60
label_freq = 10
##############################

time_running = 0
label_time = 0
label_period = 1/label_freq

draw_time = 0
draw_period = 0.1

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

last_time = time.time() # or else you do a massive dt to start
while time_running < rec_time:
    curr_time = time.time()
    dt = curr_time - last_time
    last_time = curr_time
    time_running += dt
    label_time += dt
    draw_time += dt

    Walker.slow_update(dt)
    theta = Walker.heading()

    if draw_time >= draw_period:
        draw_time = 0
        Viz.draw_heading(theta)

    if label_time >= label_period:
        print(label_time)
        label_time = 0
        File.write_heading(theta)

stream.stop_stream()
stream.close()


zero_num = get_zero_string(DATA_N)

path = "/Users/zachyamaoka/Dropbox/de3_audio_data/data_real_wav/"
file_name = "real_rec_"
name=path+file_name+zero_num+".wav"
print(name)
save_wav(frames, name)

p.terminate()
