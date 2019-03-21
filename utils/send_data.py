import argparse
import random
import time

from pythonosc import osc_message_builder
from pythonosc import udp_client

class DataSender():

    def __init__(self, ip="127.0.0.1", port=7400):
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default=ip, help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=port, help="The port the OSC server is listening on")
        args = parser.parse_args()
        self.client = udp_client.SimpleUDPClient(args.ip, args.port)

    def send_position(self, x, y):
        """Send x y positon to Sophie's computer to be displayed on projector"""
        self.client.send_message("/filter", (x,y))

    def send_bel(self, bel):
        bel = tuple(bel)
        self.client.send_message("/heading", bel)

    def send_heading(self, theta, var):
        """Send x y positon to Sophie's computer to be displayed on projector"""
        self.client.send_message("/heading", (theta,var))


#### TEST CODE ####

# Sender = DataSender()
#
# Sender.send_position(10,10)
# Sender.send_heading(10,10)
