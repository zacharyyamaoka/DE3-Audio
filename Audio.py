import pyaudio
import numpy as np

import pyaudio
import wave

from debugger import *
from audio_utils import *

class AudioPlayer():
    def __init__(self, chunk = 1024):
        # self.p = pyaudio.PyAudio()
        self.CHUNKSIZE = chunk

    def load_wav_audio(self, file):
        wf = wave.open(file, 'rb')
        p = pyaudio.PyAudio()
        self.stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

        self.wf = wf
        self.num_samples = wf.getnframes()
    def play_this(self, song):
        self.stream.write(song)

    def play(self):
        data = self.wf.readframes(self.CHUNKSIZE)
        while data != '':

            self.stream.write(data)
            data = self.wf.readframes(self.CHUNKSIZE)

    def get_all_data(self):
        data = self.wf.readframes(self.num_samples)
        sound_array = decode(data,2)
        return sound_array


    def stream_audio(self, live=False, get_data=False, use_viz=False):
        if live:
            pass
        else: #play back pre recorded
            data = self.wf.readframes(self.CHUNKSIZE)
            if data != '':
                sound_array = decode(data,2)
                self.stream.write(data)
                # if use_viz:
                #     viz(sound_array)
                # if get_data:
            else:
                self.stream.stop_stream()
                self.stream.close()

        le, re = sound_array[:,0], sound_array[:,1]
        return le, re

#how am I going to deal with nan's in the audio signal....
#real time aspect complicate things greatly, the fact that it is real audio

def normalize(data):
    pass
