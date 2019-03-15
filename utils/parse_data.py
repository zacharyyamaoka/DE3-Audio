

#This File loads in formats audio data into a array. Correlates with audio file.

import numpy as np
import pyaudio
import wave
#define params manually here
from audio_utils import *
from Audio import *


#encapsulate this all at some point
dataset = []

sample_rate = 44100 #hz
label_rate = 1 #label/sec


#careful ensure your not clipping when you max out bit depth.....


path = "data_rec"
numofrecs = 1
# load audio file. Clip of the start before first sound plays


#for File
n = 1
#load labels
zeros = "00"
chunk = 1024
label = np.loadtxt("data_label/"+path+zeros+str(n)+".txt")

file_path = "data_wav/"+path+zeros+str(n)+".wav"

player = AudioPlayer(chunk)
print(file_path)
player.load_wav_audio(file_path)
data = player.get_all_data()

#regardless clip of first 2 seconds if data
start_clip = sample_rate * 2 #seconds
data = data[start_clip:-1]
num_sample = data.shape[0]

# find start and end
start = 0
end = num_sample - 1

for i in np.arange(num_sample):
    if data[i,0] != 0:
        start = i
        break;

for i in np.arange(num_sample):
    j = num_sample - 1 - i
    if data[j,0] != 0:
        end = j
        break;

print(start, end)
audio = encode(data)
# player.play_this(audio)
data = data[start:end] #remove silence
parsed_audio = encode(data)
# player.play_this(parsed_audio)
# print("num samples ", player.num_samples)
# print(data.shape)

#process data to remove silence at start and end....


dataset.append([data,label,[sample_rate,label_rate]])
print("Shapes")
print(data.shape)
print(label.shape)

def sample2labelId(n, sample_rate,label_rate): #get label which correlates with sample
    timestamp = n/sample_rate
    print(timestamp)
    label_num = int(timestamp*label_rate)
    return label_num

ind = sample2labelId(2703871, sample_rate, label_rate) #will be some small error, but no more then 1 label so its ok....


print(ind)
# print(dataset)
# while True:
#     le, re = player.stream_audio(live=False, get_data=True, use_viz=False)
#     # if counter % 20 == 0:
#     assert chunk == le.shape[0] #make sure these are equal. if not, check your using 16 bit res audio
#     print(le.shape)
#     print(re.shape)


#! TODO creat vizulaiser for data.


# wf = wave.open(file_path, 'rb')
#
# num_samples = wf.getnframes()
# sample_width = wf.getsampwidth()
# print(sample_width)
# print(num_samples)
# sample_freq = wf.getframerate()
# time = num_samples/sample_freq
# audio_data = wf.readframes(num_samples)
# print(time)
# print(wf)
# # print(audio_data)
# audio_vector = decode(audio_data,2)
# print(audio_vector.shape)
# data = []
