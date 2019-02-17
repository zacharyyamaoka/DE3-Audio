import numpy as np

import unittest
import os
import sys

from ALA import ala

class TestMain(unittest.TestCase):

    def test_ALA(self):
        re = [0]
        le = [0]
        r, theta = ala(re,le)
        self.assertEqual(r,0)
        self.assertEqual(theta,0)

if __name__ == '__main__':
    unittest.main()
