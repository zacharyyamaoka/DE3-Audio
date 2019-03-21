
from locate import SoundLocalizer
from filter import PositionFilter
import numpy as np
import matplotlib.pyplot as plt
# Audio Localization Algorithm



class ALA():

    def __init__(self, res=20):

        self.Localizer = SoundLocalizer()
        self.DBF = PositionFilter(res)
        self.measure_var = np.pi #based on localization bin size - we just do left right


    def update(self, dt=0.1): #must call this each loop
        self.DBF.motion_update(dt)

    def new_reading(self, audio_vec, head_yaw=0): # Use CNN to locate sound, assume measurement variance here
        theta_mu, confidence = self.Localizer.locate(audio_vec) #Localize sound
        var = self.measure_var #

        theta_mu += head_yaw # shift to account for moving head

        if confidence > 0: #outlier rejection
            self.DBF.sensor_update(theta_mu, var)

    def get_best_estimate(self): # Main Computation

        return self.DBF.get_peak(), self.measure_var  #return max of distrbution and pre set variance

    def get_bel(self):
        return self.DBF.bel
    def get_bin_n(self):
        return self.DBF.n
    def get_step(self):
        return self.DBF.step
