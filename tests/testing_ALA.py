import numpy as np
import pyaudio

import unittest
import os
import sys

from ALA import ala
from Audio import *

class TestMain(unittest.TestCase):

    def test_ALA(self):
        re = [0]
        le = [0]
        r, theta = ala(re,le)
        self.assertEqual(r,0)
        self.assertEqual(theta,0)

    def test_stream_audio(self):
        pass

    # def test_imports(self):
    #     try:
    #         import tensorflow
    #     except:
    #         self.assertEqual(1,0)
if __name__ == '__main__':
    unittest.main()
