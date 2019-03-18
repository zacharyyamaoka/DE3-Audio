
from locate import SoundLocalizer
from filter import PositionFilter
import numpy as np
# Audio Localization Algorithm

Localizer = SoundLocalizer()
KF = PositionFilter()

def ala(audio_vec): # Main Computation

    pred_theta = Localizer.locate(audio_vec)
# pred_theta
    pred_var = np.pi #180

    theta_mu, theta_var = KF.filter(pred_theta, pred_var)

    # theta_var = np.pi/2
    theta_var = np.pi

    pred_theta = pred_theta % (2*np.pi)
    print("Pred Thheta: ", pred_theta)

    if pred_theta > 0 and pred_theta < np.pi:
        theta_mu = np.pi/2

    if pred_theta <= 0 and pred_theta > -np.pi:
        theta_mu = -np.pi/2

    if pred_theta > np.pi and pred_theta <= 2*np.pi:
        theta_mu = -np.pi/2

    return theta_mu, theta_var
