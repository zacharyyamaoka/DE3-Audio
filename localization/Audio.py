import pyaudio
import numpy as np

import pyaudio
import wave

from debugger import *


class AudioPlayer():
    def __init__(self):
        # self.p = pyaudio.PyAudio()
        self.CHUNKSIZE = 1024
        print(0)

    def load_wav_audio(self, file):
        wf = wave.open(file, 'rb')
        p = pyaudio.PyAudio()
        self.stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

        self.wf = wf

    def play(self):
        data = self.wf.readframes(self.CHUNKSIZE)
        while data != '':

            self.stream.write(data)
            data = self.wf.readframes(self.CHUNKSIZE)

    def stream_audio(self, live=False):
        if live:
            pass
        else: #play back pre recorded
            data = self.wf.readframes(self.CHUNKSIZE)
            if data != '':
                sound_array = decode(data,2)
                # viz(sound_array)
                self.stream.write(data)
                data = self.wf.readframes(self.CHUNKSIZE)
            else:
                self.stream.stop_stream()
                self.stream.close()

        le, re = sound_array[:,0], sound_array[:,1]
        return le, re

#how am I going to deal with nan's in the audio signal....
#real time aspect complicate things greatly, the fact that it is real audio

def normalize(data):
    pass
# Code taken from https://stackoverflow.com/questions/22636499/convert-multi-channel-pyaudio-into-numpy-array
def decode(in_data, channels):
    """
    Convert a byte stream into a 2D numpy array with
    shape (chunk_size, channels)

    Samples are interleaved, so for a stereo stream with left channel
    of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
    is ordered as [L0, R0, L1, R1, ...]
    """
    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    result = np.fromstring(in_data, dtype='Int16')

    chunk_length = len(result) / channels
    assert chunk_length == int(chunk_length)

    result = np.reshape(result, (int(chunk_length), int(channels)))
    return result


def encode(signal):
    """
    Convert a 2D numpy array into a byte stream for PyAudio

    Signal should be a numpy array with shape (chunk_size, channels)
    """
    interleaved = signal.flatten()

    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    out_data = interleaved.astype('Int16').tostring()
    return out_data
