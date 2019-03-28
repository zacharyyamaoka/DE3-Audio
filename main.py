import os
import sys

sys.path.append(os.path.join(os.getcwd(), "utils"))
sys.path.append(os.path.join(os.getcwd(), "algo"))
from timer_utils import *
from ALA import ALA
from AudioLive import *
from debugger import *
import time
from send_data import DataSender

from audio_utils import *
from dummy_head import DummyHead

plt.ion()

room_width = 5
room_length = 5

SAMPLE_RATE = 96000
# CHUNK_SIZE = 192512
WINDOW_TIME = 0.005
CHUNK_SIZE = int(SAMPLE_RATE*WINDOW_TIME)
VIZ = True
# Sender = DataSender(ip="146.169.220.251", port=7400)
Sender = DataSender(port=7400)
# Head = DummyHead(14141)
Player = LivePlayer(window = WINDOW_TIME, sample_rate=SAMPLE_RATE, playback=True)
# Viz = Debugger()
ALA = ALA()
counter = 0

#IMPORTANT
#source activate tensorflow
# player.load_wav_audio("data_wav/data_rec001.wav")

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_TIME)
#Normal Audio Signal to go between 1 and 0? so all db the same... or is it a matter or fine tunning, live calibrating.....

last_time = time.time()
send_timer = 0

theta_mu = 0
theta_var = np.pi

if VIZ:
    fig4 = plt.figure(4,figsize=(10,5))
    ax4 = fig4.add_subplot(111)

while True:
    curr_time = time.time()
    dt = curr_time - last_time
    last_time = curr_time

    send_timer += dt

    counter += 1
    # Player.stream_audio(live=True, playback=True)

    # Head.look_around()

    #TODO TUNE drift parameter
    ALA.update(dt) #call each loop
    #
    if Player.full:
        data_raw = Player.get_sample_rec() #latest files in the queue
        data_raw = b''.join(data_raw)
        data_vec = decode(data_raw,2)

        #only take the x most recent ones
        #easier to just cut to size b/c time diff is so small
        #TODO make it so it just gets the most recent one
        data_vec = data_vec[:CHUNK_SIZE]

        # head_yaw = Head.get_rotation()
        head_yaw = 0
        ALA.new_reading(data_vec, head_yaw) #update belief


    theta_mu, theta_var = ALA.get_best_estimate() #get the best estimate of where the source is each time stamp


    if VIZ:
        theta = np.arange(ALA.get_bin_n()) * ALA.get_step()
        theta += ALA.get_step()/2 #center
        theta_new = []

        for w in theta:
            if w > np.pi: #wrap back
                w = w - (2*np.pi)

            theta_new.append(w)

        #for viz purposes move to pi to =pi
        ax4.clear()
        ax4.set_ylim(0,1)
        plt.scatter(theta_new,ALA.get_bel())

        # center theta_mu

        if theta_mu > np.pi: #wrap back
            theta_mu = theta_mu - (2*np.pi)
        plt.scatter(theta_mu,0.8, s=40, c='b', marker='o')

        plt.show()
        plt.pause(0.01)

    # print(theta_mu, theta_var)
    #if very uncertain then you can have a massive variance that covers the entire circle

    if send_timer > 0.2: #every x seconds send your best estimate of theta_mu and theta_var
        send_timer = 0
        #send Prediction to Sophie
        Sender.send_bel(ALA.get_bel())
        # Sender.send_heading(float(np.rad2deg(theta_mu)),np.rad2deg(theta_var))

    continue
