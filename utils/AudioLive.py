from collections import deque
import pyaudio

class LivePlayer():

    def __init__(self, window = 0.1, sample_rate=44100, width=2, playback=False):

        self.p = pyaudio.PyAudio()

        self.FORMAT = self.p.get_format_from_width(width)
        self.CHANNELS = 2
        self.RATE = sample_rate


        self.frames = deque(maxlen = int(sample_rate*window))

        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        output=playback,
                        stream_callback=self.callback)

        # self.stream.stop_stream()

    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def get_sample(self):
        """Returns most recent samples in window segment"""
        sample = b''.join(self.frames)
        return sample
