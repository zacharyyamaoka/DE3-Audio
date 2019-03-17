from collections import deque
import pyaudio
import copy

class LivePlayer():

    def __init__(self, window = 0.1, sample_rate=44100, width=2, playback=False, record = False):

        self.p = pyaudio.PyAudio()

        self.WIDTH = width
        self.FORMAT = self.p.get_format_from_width(self.WIDTH)
        self.CHANNELS = 2
        self.RATE = sample_rate

        self.buffer_len = int(sample_rate*window)
        # self.frames = deque(maxlen = self.buffer_len)
        self.frames = []
        self.frame_count = 0
        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        output=playback,
                        stream_callback=self.callback)

        # self.stream.stop_stream()
        self.full = False

    def callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        self.frame_count += frame_count
        if self.frame_count >= self.buffer_len:
            self.full = True

        return (in_data, pyaudio.paContinue)

    def clear(self):
        self.frames.clear()
        self.frame_count = 0
        self.full = False
    def get_sample(self):
        """Returns most recent samples in window segment"""

        return self.frames

    def get_sample_rec(self):
        """returns buffer and clears it"""

        # sample = b''.join(self.frames)
        sample = copy.deepcopy(self.frames)
        self.frames.clear()
        self.frame_count = 0
        self.full = False
        return sample
