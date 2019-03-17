import numpy as np
import pyaudio
import wave
#define params manually here
from audio_utils import *
from Audio import *
from debugger import *

def get_zero_string(n):
    """Returns left padded number with zeros for file search

    Args:
        n (int): document number you want to query
    Returns:
        string: value in a left padded zero string. 1 -> 001, 10 -> 010 -> 100, 100 -> 100
    """
    zero_str = ''

    if n < 10:
        zero_str =  "00" + str(n)
    elif n < 100:
        zero_str =  "0" + str(n)
    elif n < 1000:
        zero_str = str(n)

    return zero_str


def load_data_file(n=1,audio_n_offset=0,file_stem="data_rec",sample_rate = 44100,label_rate = 1): #only works up to 10 rn
    #========================================
    #SET CORRECT PATHS
    data_label_path = "./data_label" + "/"
    data_wav_path = "./../data_wav" + "/"


    #========================================

    label_num = get_zero_string(n+audio_n_offset)
    audio_num = get_zero_string(n+audio_n_offset)

    label_file_path = data_label_path+file_stem+label_num+".txt"
    audio_file_path = data_wav_path+file_stem+audio_num+".wav"

    #First load labels from the audio file
    label = np.loadtxt(label_file_path)
    num_labels = label.shape[0]

    #load audio file
    chunk = 1024
    player = AudioPlayer(chunk)
    player.load_wav_audio(audio_file_path)
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
    #print("num labels, ", num_labels)
    label_time = (num_labels-1) * (1/label_rate) #so you get better use out of your last label
    #print("Label Time: ", label_time)
    num_sample_labeled = label_time * sample_rate
    end = int(start + num_sample_labeled)
    data = data[start:end,:]

    len_data = data.shape[0]
    print("Song time: ", len_data/sample_rate)
    #find end based on num of samples

    return [data,label,[sample_rate,label_rate]] #list of data

def listen_2_data(all_data, window_chunk=1024, audio_rate=44100, label_rate=100):
    data = all_data[0]
    label = all_data[1]
    curr_frame = 0
    last_frame = data.shape[0] - 1
    last_ind = -1
    Viz = Debugger()
    room_width = 5
    room_length = room_width

    player = AudioPlayer(window_chunk)
    count = 0

    while True:

        clip = data[curr_frame:curr_frame+window_chunk,:]
        parsed_audio = encode(clip)

        curr_frame += window_chunk
        if curr_frame > last_frame:
            break

        ind = sample2labelId(curr_frame,44100,label_rate)

        count += 1

        if count % 50 == 0:
        # if ind != last_ind: #only go further if he has moved, may get lag if label rate 2 high
            last_ind = ind
            pos = label[ind,:]
            r = np.sqrt(pos[0]**2 + pos[1]**2)
            theta = np.cos(pos[0]/r)
            #print(r, theta)
            Viz.draw_sound_in_room(pos[0], pos[1])

        player.play_this(parsed_audio)


def sample2labelId(n, sample_rate,label_rate): #get label which correlates with sample
    timestamp = n/sample_rate
    #Used to be -1 but that does not give the correct behavour
    #With audio samples you want to floor(). Any samples should correspond to the last label, not the nearest... do not round()
    label_num = int(timestamp*label_rate) #so you can index correctly in list
    return label_num


#Make function to playback music and vizualize

def show_data(data, labels):
    pass


if __name__ == '__main__':
    data = load_data_file(n=1,label_rate = 100)
    listen_2_data(data)
    #print("Data Params")
    #print("Audio Stero Vector: ", data[0].shape, " Sample Rate: ", data[2][0])
    #print("Position Label: ", data[1].shape, " Label Rate: ", data[2][1])
    '''
    ind = sample2labelId(0, 44100, 1) #will be some small error, but no more then 1 label so its ok....
    # assert ind == 0
    print(ind)
    ind = sample2labelId(data[0].shape[0], 44100, 1) #will be some small error, but no more then 1 label so its ok....
    print(ind)
    '''
