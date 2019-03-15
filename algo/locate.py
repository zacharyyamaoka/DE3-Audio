import numpy as np
class SoundLocalizer():

    def __init__(self):
        print("SoundLocalizer Init")

    def locate(self, audio_vec):

        #PUT HAROONS CODE INTO HERE

        radius =  1
        theta = 0
        re = audio_vec[0,:]
        le = audio_vec[1,:]
        r_total = np.sum(re)
        l_total = np.sum(le)

        if r_total>l_total:
            theta = 0

        if r_total<l_total:
            theta = np.deg2rad(180)

        return radius, theta
