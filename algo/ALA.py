
from locate import SoundLocalizer
from filter import PositionFilter

# Audio Localization Algorithm

Localizer = SoundLocalizer()
Filterer = PositionFilter()

def ala(audio_vec):

    r, theta = Localizer.locate(audio_vec)
    r_filtered, theta_filtered = Filterer.filter(r, theta)

    return r_filtered, theta_filtered
