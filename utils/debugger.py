import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Debugger():

    def __init__(self):
        plt.ion()
        plt.show()

        # self.fig2 = plt.figure(2)
        # self.l_ax = self.fig2.add_subplot(211)
        # self.r_ax = self.fig2.add_subplot(212)

        self.fig1 = plt.figure(1)
        self.ax1 = self.fig1.add_subplot(111, aspect='equal')
        self.w = 10


    def viz(self, sound):
        channels = sound.shape[1]
        print("DRAWING")
        self.l_ax.clear()
        self.r_ax.clear()

        self.l_ax.set_ylim(-65536,65536) # 2^16 = 65536
        self.l_ax.plot(sound[:,0])

        self.r_ax.set_ylim(-65536,65536)
        self.r_ax.plot(sound[:,1])
        plt.show()
        plt.pause(0.00001)

    def draw_head(self):
        circle= plt.Circle((0,0), radius= 0.2) #20 cm
        self.ax1.add_patch(circle)

    def draw_sound_in_room(self, x, y):
        # head = plt.Circle((0, 0), 0.2, color='r')

        #rotate so head is facing forward
        x = -y
        y = x
        self.ax1.clear()
        self.draw_head()
        self.ax1.plot(x,y,'ro')
        self.ax1.set_ylim(-self.w/2,self.w/2)
        self.ax1.set_xlim(-self.w/2,self.w/2)
        # ax.plot(w/2,l/2,'b+')
        # ax.plot(w/2,-l/2,'b+')
        # ax.plot(-w/2,l/2,'b+')
        # ax.plot(-w/2,-l/2,'b+')
        # # ax.add_artist(head)
        # # ax.set_xlim([-room_width/2,room_width/2])
        # # ax.set_ylim([-room_length/2,room_length/2])
        # room = patches.Rectangle((-w/2,-l/2),w,l,linewidth=1,fill=False)
        # ax.add_patch(room)
        plt.show()
        plt.pause(0.0001)
