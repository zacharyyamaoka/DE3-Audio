# Communicates withh 3D test app throughh OSC
import random
import time
import numpy as np
from move_utils import *
from data_writer_util import LabelFile



Controller = TuneInControl(port=12300)
Walker = RandomPolarWalker()

#MAKE SURE TO CHANGE THE FILE NUMBER
File = LabelFile(1)

last_time = time.time()

time_running = 0
rec_time = 1 * 60

label_time = 0

label_freq = 100
label_period = 1/label_freq

loop_time = 0.001
once = True
Controller.play()

while time_running < rec_time:

    curr_time = time.time()
    dt = curr_time - last_time
    last_time = curr_time
    time_running += dt
    label_time += dt
    # print(dt)
    Walker.update(dt) #pick dt so it matcehs the chunck size.... or is it the dt of the loop?
    x, y, z = Walker.location()
    Controller.send_position(x,y,z)

    if label_time >= label_period: #rate at which I write is indepent and will slightly vary.....
        print(label_time)
        # print(label_time)
        label_time = 0 #start counting agian
        File.write_pos(x,y,z)

    #wait util loop time is
    while time.time() - curr_time < loop_time:
        pass #wait untill your good to go
    # time.sleep(0.001) #essentiall I want the movment to be as smooth as possible, but within reason OSC has limited commication rate....

Controller.pause()
File.close()
