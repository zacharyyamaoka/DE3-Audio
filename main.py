import os
import sys

sys.path.append(os.path.join(os.getcwd(), "utils"))
sys.path.append(os.path.join(os.getcwd(), "algo"))

from ALA import ala
from Audio import *
from debugger import *
import time

room_width = 5
room_length = 5

Player = AudioPlayer()
Viz = Debugger()

counter = 0

#IMPORTANT
#source activate tensorflow
# player.load_wav_audio("data_wav/data_rec001.wav")


#Normal Audio Signal to go between 1 and 0? so all db the same... or is it a matter or fine tunning, live calibrating.....
while True:
    counter += 1
    audio_vec = Player.stream_audio(live = True, playback=True)

    r, theta = ala(audio_vec)
    print(counter)
    if counter % 20 == 0:
        pass
        # Viz.draw_sound_in_room(room_width,room_length,r,theta)
