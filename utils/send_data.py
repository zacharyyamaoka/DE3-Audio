import argparse
import random
import time

from pythonosc import osc_message_builder
from pythonosc import udp_client

class DataSender():

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--ip", default="146.169.222.168", help="The ip of the OSC server")
        parser.add_argument("--port", type=int, default=7400, help="The port the OSC server is listening on")
        args = parser.parse_args()
        self.client = udp_client.SimpleUDPClient(args.ip, args.port)

    def send_position(self, x, y):
        """Send x y positon to Sophie's computer to be displayed on projector"""
        self.client.send_message("/filter", (x,y))



#### TEST CODE ####

Sender = DataSender()

Sender.send_position(10,10)
