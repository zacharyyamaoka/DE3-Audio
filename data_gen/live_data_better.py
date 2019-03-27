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
from data_writer_util import *
from data_utils import get_zero_string
from AudioLive import LivePlayer


# N = 0
# nbefore tunning 2106
file_count = 3000 #enter nexts file num
WINDOW_TIME = 2
rec_min = 0.5
SAMPLE_RATE = 96000
rec_time = rec_min * 60

Player = LivePlayer(window = WINDOW_TIME, playback=True, sample_rate=SAMPLE_RATE)
wav = WavWriter(path="/Users/zachyamaoka/Dropbox/de3_audio_data/data_clip/", rate=SAMPLE_RATE)
Walker = RandomPolarWalker(rec_time)
Viz = Debugger()
# file = BatchLabel(name="label_"+str(N),path="/Users/zachyamaoka/Documents/de3_audio/data_clip_label/")
file = BatchLabel(name="label",path="/Users/zachyamaoka/Documents/de3_audio/data_clip_label/")

#MAKE SURE TO CHANGE THE FILE NUMBER

last_time = time.time()

##############################
"""IMPORTANT"""
label_freq = 10
count_down = 4

##############################


time_running = 0
label_time = 0
label_period = 1/label_freq

draw_time = 0
draw_period = 0.1

# Wait for person to get into correct position
theta = Walker.heading()

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
Player.clear()
last_window = time.time()
while time_running < rec_time:
    curr_time = time.time()
    dt = curr_time - last_time
    last_time = curr_time
    time_running += dt
    label_time += dt
    draw_time += dt

    # theta = Walker.heading()
    # print(dt)
    # print(len(Player.frames))
    if Player.full and curr_time - last_window >= WINDOW_TIME: #buffer full
        last_window = time.time()
        print("Saving Files: time stamp ", time_running)
        data = Player.get_sample_rec()
        # print(data)
        # wav.save_wav("clip"+str(file_count), data)

        theta = Walker.heading()
        Walker.slow_update(WINDOW_TIME)
        Viz.draw_heading(theta,show = True)

        # file.write("clip"+str(file_count)+".wav", theta)
        file_count += 1
    #
    # if draw_time >= draw_period:
    #     draw_time = 0
    #     Viz.draw_heading(theta)


file.close()
