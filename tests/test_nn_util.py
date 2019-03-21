import unittest
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "nn_utils"))
sys.path.append(os.path.join(os.getcwd(), "solver"))
import numpy as np
from nn_util import *

class TestMain(unittest.TestCase):
    def test_get_theta_quad(self):

        self.assertEqual(get_theta_quad(0, 4),np.pi/2)
        self.assertEqual(get_theta_quad(3, 4)%(2*np.pi),0)

    def test_get_bins(self):

        n = 4
        b = get_bins(n)
        self.assertEqual(len(b),n)

        self.assertEqual(get_bins(2)[0],0) #start at zero for left or right
        self.assertEqual(get_bins(4)[0],np.deg2rad(45)) #shhouldn't start at zerot
        # self.assertTrue(get_bins(5))
        if False:
            x = []
            y = []
            r = 1
            for theta in b:
                x.append(r * np.cos(theta))
                y.append(r * np.sin(theta))

            plt.plot(y, x, '+')
            plt.show()
    def test_segment_data(self):

        b = get_bins(2)

        label = segment_data(np.pi/2,b)

        self.assertEqual(label,0) #start at zero for left or right

        label = segment_data(1.5*np.pi,b)

        self.assertEqual(label,1)

        b = get_bins(4)

        label = segment_data(np.pi/2,b)
        self.assertEqual(label,0)
        label = segment_data(0,b)
        self.assertEqual(label,3)

if __name__ == '__main__':
    unittest.main()
