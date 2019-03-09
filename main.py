from ALA import ala
from Audio import *
from debugger import *
import time

room_width = 5
room_length = 5

player = AudioPlayer()
# player.load_wav_audio("audio_files/binuaral_test.wav")
player.load_wav_audio("audio_files/test_2.wav")
player.play()
while True:
    le, re = player.stream_audio()

counter = 0


#IMPORTANT
#source activate tensorflow

while True:
    counter += 1

    le, re = player.stream_audio()
    print(len(le))
    #print(len(re))
    #time.sleep(1)
    r, theta = ala(le, re)

    if counter % 20 == 0:
        draw_sound_in_room(room_width,room_length,r,theta)
