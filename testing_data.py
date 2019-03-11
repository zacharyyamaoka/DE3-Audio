from ALA import ala
from Audio import *
from debugger import *
import time

room_width = 5
room_length = 5

player = AudioPlayer()
# player.load_wav_audio("audio_files/binuaral_test.wav")
player.load_wav_audio("audio_files/Ciao006.wav")
data =  player.get_data()
viz(sound_array)
print(data)
