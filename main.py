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

SAMPLE_RATE = 44100
WINDOW_TIME = 0.0001
Sender = DataSender()
Player = LivePlayer(window = WINDOW_TIME, playback=True)
Viz = Debugger()

counter = 0

#IMPORTANT
#source activate tensorflow
# player.load_wav_audio("data_wav/data_rec001.wav")

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_TIME)
#Normal Audio Signal to go between 1 and 0? so all db the same... or is it a matter or fine tunning, live calibrating.....
tic()
while True:
    toc()
    tic()
    counter += 1
    # Player.stream_audio(live=True, playback=True)

    #Grad Recording
    data_raw = Player.get_sample()
    data_vec = decode(data_raw,2)

    # Get Prediction
    theta_mu, theta_var = ala(data_vec)
    print(theta_mu,theta_var)

    #send Prediction to Sophie
    Sender.send_heading(theta_mu,theta_var)
