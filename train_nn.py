import librosa
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

#audio, sr = librosa.load('./audio_files/binaural_test.wav', sr=44100, mono=False)

class AudioLocationDataset(Dataset):
    def __init__(self, root="./audio_files/", csv="./labels.csv", transform=None):
        self.root = root
        self.csv = pd.read_csv(csv)
        #self.filenames = self.csv['Filename'].tolist()
        #self.labels = self.csv['Label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        audiopath = self.root+self.csv.iloc[idx, 0]
        audio, sr = librosa.load(audiopath, sr=44100, mono=False)
        label = eval(self.csv.iloc[idx, 1].replace('/', ','))

        if self.transform:
            audio = self.transform((audio, label))

        return audio, label

class ToTensor():
    def __init__(self):
        pass

    def __call__(self, sample):
        a, l = sample
        return torch.Tensor(a), torch.Tensor(l)

class AudioLocationNN(torch.nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass


data = AudioLocationDataset(transform=ToTensor())




