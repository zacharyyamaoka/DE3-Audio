import os
import sys

sys.path.append(os.path.join(os.getcwd(), "utils"))
sys.path.append(os.path.join(os.getcwd(), "algo"))
from timer_utils import *
from ALA import ala
from AudioLive import *
from debugger import *
import time
from send_data import DataSender
from audio_utils import *

room_width = 5
room_length = 5

SAMPLE_RATE = 96000
CHUNK_SIZE = 192512
WINDOW_TIME = CHUNK_SIZE/float(SAMPLE_RATE)

# Sender = DataSender(ip="146.169.222.244", port=7400)
Sender = DataSender(port=7400)

Player = LivePlayer(window = WINDOW_TIME, sample_rate=SAMPLE_RATE, playback=True)
Viz = Debugger()

counter = 0

#IMPORTANT
#source activate tensorflow
# player.load_wav_audio("data_wav/data_rec001.wav")

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_TIME)
#Normal Audio Signal to go between 1 and 0? so all db the same... or is it a matter or fine tunning, live calibrating.....

last_time = time.time()
send_timer = 0

while True:
    curr_time = time.time()
    dt = curr_time - last_time
    last_time = curr_time

    send_timer += dt

    counter += 1
    # Player.stream_audio(live=True, playback=True)

    #Grad Recording
    if Player.full:
        data_raw = Player.get_sample_rec() #latest files in the queue
        data_raw = b''.join(data_raw)
        data_vec = decode(data_raw,2)

        #only take the x most recent ones
        #easier to just cut to size
        data_vec = data_vec[:CHUNK_SIZE]
        print(data_vec.shape)
        # Get Prediction
        theta_mu, theta_var = ala(data_vec)
        print(theta_mu,theta_var)

        if send_timer > 0.2:
            print("sending!", send_timer)
            send_timer = 0
            #send Prediction to Sophie
            Sender.send_heading(np.rad2deg(theta_mu),np.rad2deg(theta_var))
    else:
        continue # still waiting for data
