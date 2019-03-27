import unittest
import numpy as np
import os
import sys
import time
sys.path.append(os.path.join(os.getcwd(), "data_gen"))
sys.path.append(os.path.join(os.getcwd(), "utils"))
from debugger import *
import matplotlib.pyplot as plt
from move_utils import RandomPolarWalker
class TestMain(unittest.TestCase):

    def test_RandomPolarWalker(self):

        plt.ion()
        plt.show()

        fig1 = plt.figure(1,figsize=(16,9))
        ax1 = fig1.add_subplot(111)
        ax1.set_ylim(-5,5)
        ax1.set_xlim(-5,5)

        x_path = []
        y_path = []

        curr_time = time.time()

        time_running = 0
        rec_time = 0.5 * 60

        walker = RandomPolarWalker()

        while time_running < rec_time:
            print("running")
            dt = time.time() - curr_time
            curr_time = time.time()
            time_running += dt

            walker.update(dt)
            x, y, z = walker.location()
            x_path.append(x)
            y_path.append(y)
            ax1.plot(y,x,'+')
            plt.pause(0.01)

    # print(x_path)
    # plt.plot(y_path,x_path,'+')
    # plt.show()
    # def test_RandomPolarWalker(self):
    #
    #     Viz = Debugger()
    #     x_path = []
    #     y_path = []
    #
    #     curr_time = time.time()
    #
    #     time_running = 0
    #     rec_time = 5 * 60
    #
    #     walker = RandomPolarWalker()
    #
    #     while time_running < rec_time:
    #         print("running")
    #         dt = time.time() - curr_time
    #         curr_time = time.time()
    #         time_running += dt
    #
    #         walker.slow_update(dt)
    #         x, y, z = walker.location()
    #         heading = walker.heading()
    #         Viz.draw_heading(heading)
    #
    #     # print(x_path)
    #     # plt.plot(y_path,x_path,'+')
    #     # plt.show()

if __name__ == '__main__':
    unittest.main()
