import unittest
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "algo"))

import numpy as np
from filter import *

class TestMain(unittest.TestCase):

    def test_PositionFilter(self):

        filter = PositionFilter()
        self.assertEqual(np.sum(filter.bel),1)

    def test_update(self):
        filter = PositionFilter()

        plt.ion()


        # filter.update(np.pi/2,np.pi) #left side
        filter.update(0,np.pi) #in front # i need to implment angular distance correctly
        theta = np.arange(filter.n) * filter.step

        def show_bel():
            ax = plt.gca()
            ax.clear()
            ax.set_ylim(0,1)
            plt.scatter(theta,filter.bel)
            plt.show()
            plt.pause(0.1)


        show_bel()
        print(filter.bel)
        # plt.pause(1)

        if True :
             #sensor
            for i in range(20):
                filter.update(1.5*np.pi,np.pi) #right side
                show_bel()

            #motion
            for i in range(20):
                filter.motion_update() #right side
                show_bel()

            for i in range(100):
                filter.update(np.pi/2,np.pi) #left side
                show_bel()

            for i in range(100):
                filter.update(1.5*np.pi,np.pi) #right side
                show_bel()

    def test_new_loss(self):

        def angle_delta(x,y):
            # move into correct range
            # x = x % (2*np.pi)
            # y = y % (2*np.pi)
            #angles start in righht range
            x = np.abs(x - y)
            x = np.abs(x - np.pi)
            x = np.pi - x
            return x

        t1 = 0
        t2 = 0
        self.assertEqual(angle_delta(t1,t2),0)
        t1 = np.pi
        t2 = 0
        d = angle_delta(t1,t2)
        print(t1, t2, ":" , d)
        self.assertEqual(d,np.pi)
        t1 = -np.pi
        t2 = 0
        self.assertEqual(angle_delta(t1,t2),np.pi)
        t1 = 0
        t2 = 2*np.pi
        self.assertEqual(angle_delta(t1,t2),0)
        t1 = 0
        t2 = -2*np.pi
        self.assertEqual(angle_delta(t1,t2),0)

if __name__ == '__main__':
    unittest.main()
