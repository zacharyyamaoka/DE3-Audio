import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


plt.ion()
plt.show()
fig2 = plt.figure(2)
l_ax = fig2.add_subplot(211)
r_ax = fig2.add_subplot(212)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, aspect='equal')

def viz(sound):
    channels = sound.shape[1]
    print("DRAWING")
    l_ax.clear()
    r_ax.clear()

    l_ax.set_ylim(-65536,65536) # 2^16 = 65536
    l_ax.plot(sound[:,0])

    r_ax.set_ylim(-65536,65536)
    r_ax.plot(sound[:,1])
    plt.show()
    plt.pause(0.0000001)


def draw_sound_in_room(w,l,r,theta):
    # head = plt.Circle((0, 0), 0.2, color='r')
    ax1.clear()
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    ax1.plot(x,y,'ro')
    ax1.set_ylim(-l/2,l/2)
    ax1.set_xlim(-w/2,w/2)
    # ax.plot(w/2,l/2,'b+')
    # ax.plot(w/2,-l/2,'b+')
    # ax.plot(-w/2,l/2,'b+')
    # ax.plot(-w/2,-l/2,'b+')
    # # ax.add_artist(head)
    # # ax.set_xlim([-room_width/2,room_width/2])
    # # ax.set_ylim([-room_length/2,room_length/2])
    # room = patches.Rectangle((-w/2,-l/2),w,l,linewidth=1,fill=False)
    # ax.add_patch(room)
    plt.draw()
    plt.pause(0.00001)
