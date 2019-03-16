# Communicates withh 3D test app throughh OSC
import random
import time
import numpy as np
from move_utils import *
from data_writer_util import LabelFile
from debugger import Debugger
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "utils"))
from Audio import AudioPlayer

Walker = RandomPolarWalker()
Viz = Debugger()
#MAKE SURE TO CHANGE THE FILE NUMBER
File = LabelFile(6)

last_time = time.time()

time_running = 0
rec_time = 1 * 60

label_time = 0
label_freq = 1
label_period = 1/label_freq

loop_time = 0.001
once = True

Player = AudioPlayer()
Viz = Debugger()

while time_running < rec_time:

    curr_time = time.time()
    dt = curr_time - last_time
    last_time = curr_time
    time_running += dt
    label_time += dt

    Player.record(playback=True)


    Walker.slow_update(dt) #pick dt so it matcehs the chunck size.... or is it the dt of the loop?
    theta = Walker.heading()
    Viz.draw_heading(theta)
    print(dt)
    if label_time >= label_period: #rate at which I write is indepent and will slightly vary.....
        # print(label_time)
        # print(label_time)
        # print(label_time)
        label_time = 0 #start counting agian
        File.write_heading(theta)
    #
    # #wait util loop time is
    # while time.time() - curr_time < loop_time:
    #     pass #wait untill your good to go
    # # time.sleep(0.001) #essentiall I want the movment to be as smooth as possible, but within reason OSC has limited commication rate....


path = "/Users/zachyamaoka/Dropbox/de3_audio_data/data_wav_5/"
file_name = "real_rec_001.wav"
Player.save_rec(name=path+file_name)
# File.close()
