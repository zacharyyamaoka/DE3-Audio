
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
    num_labels = label.shape[0]
    #vars for vizualize
    room_width=5
    room_length=5

    #load audio file
    file_path = "data_wav/"+file_stem+zeros+str(n)+".wav"

    player = AudioPlayer(chunk)
    player.load_wav_audio(file_path)

    raw_data = player.get_all_data()

    #regardless clip of first 2 seconds if data #b/c small not meaningful start at the begging
    start_clip = sample_rate * 2 #seconds
    data = raw_data[start_clip:-1]
    num_sample = data.shape[0]

    # find start and end
    start = 0
    for i in np.arange(num_sample):
        if data[i,0] != 0:
            start = i
            break;
    print("num labels, ", num_labels)
    label_time = num_labels * label_rate
    num_sample_labeled = label_time * sample_rate
    end = start + num_sample_labeled
    data = data[start:end,:]
    #find end based on num of samples
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

    #Used to be -1 but that does not give the correct behavour
    #With audio samples you want to floor(). Any samples should correspond to the last label, not the nearest... do not round()
    label_num = int(timestamp*label_rate) #so you can index correctly in list
    return label_num


#Make function to playback music and vizualize

def show_data(data, labels):
    pass
'''
data = load_data_file(0,show=False)
print("Data Params")
print("Audio Stero Vector: ", data[0].shape, " Sample Rate: ", data[2][0])
print("Position Label: ", data[1].shape, " Label Rate: ", data[2][1])

ind = sample2labelId(0, 44100, 1) #will be some small error, but no more then 1 label so its ok....
# assert ind == 0
print(ind)
ind = sample2labelId(data[0].shape[0], 44100, 1) #will be some small error, but no more then 1 label so its ok....
print(ind)
'''
