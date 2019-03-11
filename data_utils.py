
import numpy as np
import pyaudio
import wave
#define params manually here
from audio_utils import *
from Audio import *
from debugger import *

def load_data_file(n=1,file_stem="data_rec",sample_rate = 44100,label_rate = 1, playback=False, show=False): #only works up to 10 rn

    #First load labels from the audio file
    zeros = "00"
    chunk = 1024
    label = np.loadtxt("data_label/"+file_stem+zeros+str(n)+".txt")

    #vars for vizualize
    room_width=5
    room_length=5

    #load audio file
    file_path = "data_wav/"+file_stem+zeros+str(n)+".wav"

    player = AudioPlayer(chunk)
    player.load_wav_audio(file_path)

    raw_data = player.get_all_data()

    #regardless clip of first 2 seconds if data
    start_clip = sample_rate * 2 #seconds
    data = raw_data[start_clip:-1]
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

    data = data[start:end] #remove silence
    if playback: #for debugging, then play back data to make sure it is valid
        print("Playback Audio")
        parsed_audio = encode(data)
        player.play_this(parsed_audio)
        print("Done.")
    if show: #vizualize guy moving on screen aswell
        curr_frame = 0
        last_ind = -1
        while True:

            clip = data[curr_frame:curr_frame+player.CHUNKSIZE,:]
            parsed_audio = encode(clip)
            # player.load_wav_audio(file_path) #load song agian
            # player.stream_audio() #loads 1024 chunks
            curr_frame += player.CHUNKSIZE
            ind = sample2labelId(curr_frame,44100,1)

            if ind != last_ind: #only go further if he has moved, may get lag if label rate 2 high
                last_ind = ind
                pos = label[ind,:]
                r = np.sqrt(pos[0]**2 + pos[1]**2)
                theta = np.cos(pos[0]/r)
                draw_sound_in_room(room_width,room_length, r, theta)

            player.play_this(parsed_audio)

    return [data,label,[sample_rate,label_rate]] #list of data

def sample2labelId(n, sample_rate,label_rate): #get label which correlates with sample
    timestamp = n/sample_rate
    print(timestamp)
    label_num = int(round(timestamp*label_rate)) - 1 #so you can index correctly in list
    return label_num


#Make function to playback music and vizualize

def show_data(data, labels):
    pass

data = load_data_file(1,show=True)
print("Data Params")
print("Audio Stero Vector: ", data[0].shape, " Sample Rate: ", data[2][0])
print("Position Label: ", data[1].shape, " Label Rate: ", data[2][1])

ind = sample2labelId(2703871, 44100, 1) #will be some small error, but no more then 1 label so its ok....
