"""Small example OSC client

This program sends 10 random values between 0.0 and 1.0 to the /filter address,
waiting for 1 seconds between each value.
"""
import argparse
import random
import time
import sys
import os
import traceback
import optparse
import logging
import math
import pandas as pd

from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

def moving_average(gain, a, b, c, d):
	#gain = (gain, a, b, c, d)
	if a == 0:
		a = gain
	elif b == 0:
		b = gain
	elif c == 0:
		c = gain
	else:
		a = b
		b = c
		c = d
		d = 0.25*(a + b + c + gain)
	return (float(a), float(b), float(c), float(d))


def feedback(message):
	if __name__ == "__main__":
	  parser = argparse.ArgumentParser()
	  parser.add_argument("--ip", default="192.168.0.8",
	      help="The ip of the OSC server")
	  parser.add_argument("--port", type=int, default=7402,
	      help="The port the OSC server is listening on")
	  args = parser.parse_args()

	  client = udp_client.SimpleUDPClient(args.ip, args.port)

	  for x in range(10):

	    client.send_message("/filter", message)
	    time.sleep(1)


"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""

def print_gain(unused_addr, args, A, B, C, D):
	average = moving_average(args, A, B, C, D)
	feedback(average)
	return args

def print_volume_handler(unused_addr, args, volume):
  print("[{0}] ~ {1}".format(args[0], volume))

def print_compute_handler(unused_addr, args, volume):
  try:
    print("[{0}] ~ {1}".format(args[0], args[1](volume)))
  except ValueError: pass
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="192.168.0.8", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=7401, help="The port to listen on")
  args = parser.parse_args()
  dispatcher = dispatcher.Dispatcher()
  y = dispatcher.map("/filter", print_gain)
  dispatcher.map("/volume", print_volume_handler, "Volume")
  dispatcher.map("/logvolume", print_compute_handler, "Log volume", math.log)
  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  #print(r)
  #print(RGB)
  #print("Serving on {}".format(server.server_address))
  server.serve_forever()


  #print(spectrum)
