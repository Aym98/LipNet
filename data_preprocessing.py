import numpy as np
import cv2 as cv
import imageio
import matplotlib.pyplot as plt
import gdown
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import rgb_to_grayscale
from torchnlp.encoders import LabelEncoder 
from torcheval.metrics.functional import word_error_rate

path = "data/s1"
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

def get_filenames(path):
    for _, _, file in os.walk(path):
        files = file

    return [file.split(".")[0].encode("utf-8") for file in files] 

def get_split_files(path, train_size = 0.9, split = "train"):
    filenames = get_filenames(path)
    training_len = int(train_size * len(filenames))
    if split == "train":
        return filenames[0 : training_len]
    else : 
        return filenames[training_len : ]

def capture_frames(video_path):
    capture = cv.VideoCapture(video_path)
    frames = []
    number_of_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    for _ in range(number_of_frames):
        ret, frame = capture.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(frame[190 : 236, 100 : 200])
    capture.release()
    return np.array(frames)
    
def reduce_frames(frames):
    mean = np.mean(frames)
    std = np.std(frames)
    return torch.from_numpy((frames - mean) / std)

def get_video_path(filename) : 
    
    filename = filename.decode("utf-8")
    return os.path.join(path, f"{filename}.mpg")
    
    

def load_video(filename, from_path = False):
    if from_path:
        return reduce_frames(capture_frames(filename))
    video_path = get_video_path(filename)
    frames = capture_frames(video_path)
    return reduce_frames(frames)
    
    
def get_annotation_path(filename) : 
    filename = filename.decode("utf-8")
    return os.path.join("data", "align\s1", f"{filename}.align")

def split_line(annotation_path):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    token = []
    for line in lines:
        words = line.split()
        if words[2] != "sil":
            token = [*token, " ", words[2]]
    return token


def split_words(line):
    tokens = []
    for word in line:
        for character in word:
            tokens.append(character)
    return tokens
    

def load_annot(filename, from_path = False) : 
    if from_path:
        return encode(split_words(split_line(filename)))
    annotation_path = get_annotation_path(filename)
    line_split = split_line(annotation_path)
    tokens = split_words(line_split)
    return encode(tokens)


def encode(data):
    encoder = LabelEncoder(vocab, reserved_labels=['...'], unknown_index=-1)
    return encoder.batch_encode(data)

def decode(data):
    encoder = LabelEncoder(vocab, reserved_labels=['...'], unknown_index=-1)
    return encoder.batch_decode(data)
    

def ctc_decode(encoded_sequence, blank=0, max_iter=2):
    decoded_sequence = []
    prev_token = blank
    iter = 0
 
    for token in encoded_sequence:
        print(token)
        if (token!= prev_token and token != blank) or (iter > max_iter and token != blank):
            prev_token = token
            decoded_sequence.append(token)
            iter = 0
        if token == prev_token:
            iter += 1
    
    return torch.tensor(decoded_sequence)
            

class LipDataset(Dataset):
    def __init__(self, path, split, train_size = 0.9):
        super(LipDataset, self).__init__()
        self.filenames = get_split_files(path, train_size=train_size, split=split)
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        frames  = load_video(filename).float()
        annot = load_annot(filename).float()
        frames = frames.unsqueeze(0)
        frames = frames.permute(1, 2, 3, 0)
        return (frames, annot.to(torch.long))
    


def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch dynamically to the 
    maximum sequence length in the batch for both data and labels.
    
    Args:
        batch: List of tuples (data, label) where `data` and `label` are tensors.

    Returns:
        Tuple of padded data and padded labels.
    """
    # Unzip the batch into separate data and label lists
    batch_data, batch_labels = zip(*batch)

    # Dynamically determine the maximum label length in the batch
    max_label_length = max(label.shape[0] for label in batch_labels)

    # Pad input data sequences
    padded_data = pad_sequence(batch_data, batch_first=True, padding_value=0)  # Pad with 0 for data

    # Pad label sequences to max_label_length
    padded_labels = pad_sequence(
        batch_labels, batch_first=True, padding_value=-1  # Pad with -1 for labels
    )

    return padded_data, padded_labels
