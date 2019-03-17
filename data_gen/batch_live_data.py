import pyaudio
import wave
import time
import calendar
import os
from threading import Thread
from collections import deque
import copy
import info

# Output audio device info
#d = info.devices()
#d.list_devices()

# Clean all old wave files
r = os.listdir('/home/mohit/Music/')
for i in r:
    if i.endswith('wav'):
        os.remove(i)

# Recording flag & file count
recording = True
idx = 0

# Setup format info
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
file_len = 1
chunk = RATE / 100

# Open initial temp file and setup
wf = wave.open('/home/mohit/Music/temp.wav', 'wb')
wf.setnchannels(1)
wf.setframerate(RATE)
wf.setsampwidth(2)

# Create fast I/O buffer
d = deque(maxlen=RATE / chunk)

# Thread called file I/O function
def write_to_file(arg):
    global wf, idx

    # Write 1s of audio to file
    for elem in arg:
        wf.writeframes(elem)

    # If file is at the desired length close it, rename it to its utc start time (cant get ms?) and open the next temp
    # file for writing to
    if wf.tell() == RATE * file_len:
        wf.close()
        os.rename('/home/mohit/Music/temp.wav', '/home/mohit/Music/'+str(calendar.timegm(time.gmtime()) - file_len) + '.wav')
        wf = wave.open('/home/mohit/Music/temp.wav', 'wb')
        wf.setnchannels(1)
        wf.setframerate(RATE)
        wf.setsampwidth(2)
        idx += 1
        print 'File created: ' + str(calendar.timegm(time.gmtime()) - file_len) + '.wav'


# Initialize pyaudio
p = pyaudio.PyAudio()

# Pyaudio callback which appends HW audio buffer data to fast I/O 1s long buffer
def callback(in_data, frame_count, time_info, status):
    if status != 0:
        print "Non zero status!!!!!!!!!!"
        exit()
    global d
    d.append(in_data)

    # If 1s worth of audio is collected, copy to secondary buffer and pass to thread function for file I/O, then
    # clear 1s buffer
    if len(d) == RATE / chunk:
        frames = copy.copy(d)
        thread = Thread(target=write_to_file, args=[frames])
        thread.start()
        d.clear()
        print 'Copied 1s buffer: '

    return in_data, pyaudio.paContinue


# Setup audio input stream
stream = p.open(input_device_index=3,
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=chunk,
                stream_callback=callback)

# Start audio input stream
stream.start_stream()

# Capture until we have X number of files for testing
while recording:
    time.sleep(0.1)
    if idx > 64:
        recording = False

stream.stop_stream()
stream.close()
wf.close()

p.terminate()
os.remove('/home/mohit/Music/temp.wav')
