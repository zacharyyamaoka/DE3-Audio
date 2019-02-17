from locate import SoundLocalizer
from filter import PositionFilter
# Audio Localization Algorithm

Localizer = SoundLocalizer()
Filterer = PositionFilter()

def ala(re,le):

    r, theta = Localizer.locate(re,le)
    r_filtered, theta_filtered = Filterer.filter(r, theta)

    return r_filtered, theta_filtered
