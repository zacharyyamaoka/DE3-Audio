
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from ALA import ala
from Audio import *
import matplotlib.patches as patches

plt.ion()
fig = plt.figure(1)
ax = fig.add_subplot(111, aspect='equal')

room_width = 5
room_length = 5

player = AudioPlayer()
player.load_wav_audio("audio_files/binuaral_test.wav")

while True:

    le, re = player.stream_audio()
    r, theta = ala(le, re)

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    head = plt.Circle((0, 0), 0.2, color='r')

    ax.plot(x,y,'ro')
    ax.plot(room_width/2,room_length/2,'b+')
    ax.plot(room_width/2,-room_length/2,'b+')
    ax.plot(-room_width/2,room_length/2,'b+')
    ax.plot(-room_width/2,-room_length/2,'b+')
    # ax.add_artist(head)
    # ax.set_xlim([-room_width/2,room_width/2])
    # ax.set_ylim([-room_length/2,room_length/2])
    room = patches.Rectangle((-room_width/2,-room_length/2),room_width,room_length,linewidth=1,fill=False)
    ax.add_patch(room)
    plt.draw()
    # plt.pause(0.00001)
    ax.clear()
