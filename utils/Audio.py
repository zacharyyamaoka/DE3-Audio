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
        self.p = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100

        self.stream_out = self.p.open(format=self.FORMAT,
                channels=self.CHANNELS ,
                rate=self.RATE,
                output=True)

        self.stream_in = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS ,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNKSIZE)

        self.frames =[]
    def load_wav_audio(self, file):
        wf = wave.open(file, 'rb')
        self.wf = wf
        self.num_samples = wf.getnframes()

    # def start_live_rec(self, chunksize = 1024):

    def play_this(self, song):
        self.stream_out.write(song)

    def play(self):
        data = self.wf.readframes(self.CHUNKSIZE)
        while data != '':

            self.stream_out.write(data)
            data = self.wf.readframes(self.CHUNKSIZE)

    def get_all_data(self):
        data = self.wf.readframes(self.num_samples)
        sound_array = decode(data,2)
        return sound_array

    def get_live_audio(self):
        data = self.stream_in.read(self.CHUNKSIZE)
        return data

    def get_live_chunk(self, chunk=1024): # for running on API
        self.stream_in.start_stream()
        data = self.stream_in.read(chunk)
        self.stream_in.stop_stream() #pause so you don't over flow

    def stream_audio(self, live=False, playback=False, get_data=False, use_viz=False):
        if live:
            data = self.get_live_audio()
        else: #play back pre recorded
            data = self.wf.readframes(self.CHUNKSIZE)

        if data != '':
            sound_array = decode(data,2)
        else:
            self.stream_out.stop_stream()
            self.stream_out.close()

        if use_viz:
            viz(sound_array)
        if get_data:
            pass
        if playback:
            self.stream_out.write(data)

        # le, re = sound_array[:,0], sound_array[:,1]
        return sound_array

    def record(self, chunk=2**12, playback=False):
        print("rec time: ", chunk/44100)
        # self.stream_in.start_stream()
        data = self.stream_in.read(chunk)
        if playback:
            self.stream_out.write(data)
        # self.stream_in.stop_stream() #pause so you don't over flow
        self.frames.append(data)

    def save_rec(self, name = "test_live_rec.wav"):

        self.stream_in.stop_stream()
        self.stream_in.close()
        self.p.terminate()
        wf = wave.open(name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

#how am I going to deal with nan's in the audio signal....
#real time aspect complicate things greatly, the fact that it is real audio

def normalize(data):
    pass
