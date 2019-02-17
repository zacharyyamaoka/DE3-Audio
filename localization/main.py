from ALA import ala
from Audio import *
from debugger import *

room_width = 5
room_length = 5

player = AudioPlayer()
player.load_wav_audio("audio_files/binuaral_test.wav")
counter = 0

while True:
    counter += 1

    le, re = player.stream_audio()
    r, theta = ala(le, re)

    if counter % 10 == 0:
        draw_sound_in_room(room_width,room_length,r,theta)
