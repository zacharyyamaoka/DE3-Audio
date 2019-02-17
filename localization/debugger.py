import numpy as np
import matplotlib.pyplot as plt


plt.ion()
fig = plt.figure(2)
l_ax = fig.add_subplot(211)
r_ax = fig.add_subplot(212)

def viz(sound):
    channels = sound.shape[1]

    l_ax.clear()
    r_ax.clear()

    l_ax.set_ylim(-65536,65536) # 2^16 = 65536
    l_ax.plot(sound[:,0])

    r_ax.set_ylim(-65536,65536)
    r_ax.plot(sound[:,1])
    plt.show()
    plt.pause(0.0000001)
