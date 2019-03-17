
from locate import SoundLocalizer
from filter import PositionFilter
import numpy as np
# Audio Localization Algorithm

Localizer = SoundLocalizer()
KF = PositionFilter()

def ala(audio_vec): # Main Computation

    pred_theta = Localizer.locate(audio_vec)

    pred_var = 2*np.pi #180

    theta_mu, theta_var = KF.filter(pred_theta, pred_var)

    return theta_mu, theta_var
