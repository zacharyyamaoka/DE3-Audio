import numpy as np
import time
class SoundLocalizer():

    def __init__(self):
        print("SoundLocalizer Init")

    def locate(self, audio_vec):
        # print("SHHAPE", audio_vec.shape)
        #PUT HAROONS CODE INTO HERE
        audio_vec = audio_vec.T

        # time.sleep(2)

        radius =  1
        re = audio_vec[0,:]
        le = audio_vec[1,:]

        r_total = np.sum(re)
        l_total = np.sum(le)

        if r_total>l_total:
            theta = np.pi/2

        if r_total<=l_total:
            theta = -np.pi/2

        # theta = np.random.uniform(-np.pi, np.pi)
        theta = -np.pi/2
        return theta
