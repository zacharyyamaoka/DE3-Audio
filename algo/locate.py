import numpy as np
class SoundLocalizer():

    def __init__(self):
        print("SoundLocalizer Init")

    def locate(self, audio_vec):

        #PUT HAROONS CODE INTO HERE

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
