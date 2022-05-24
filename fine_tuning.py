import argparse
import csv
import os

import torch
from torch.utils.data import Dataset
import configs
import librosa
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--dropout_rate', type=str, default='0.5')

args = parser.parse_args()
config = configs.CONFIG_MAP[args.config]
dropout_rate = float(args.dropout_rate)

sample_rate = 8000
lared_dir = 'lared'
dataset_dir = 'dataset'

# Load model
model = config['model'](dropout_rate=dropout_rate, linear_layer_size=config['linear_layer_size'],
                        filter_sizes=config['filter_sizes'])
checkpoint = torch.load('./checkpoints/in_use/resnet_with_augmentation/best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

Y_train = pd.read_csv('dataset/timestamps_train.csv', usecols=['Label'])
Y_test = pd.read_csv('dataset/timestamps_test.csv', usecols=['Label'])
Y_eval = pd.read_csv('dataset/timestamps_eval.csv', usecols=['Label'])


# class LaRedDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#
#
#     def __len__(self):
#
#
#     def __getitem__(self, idx):

def load_dataset():
    # TODO: Add speech data timestamps to dataset
    X_train_timestamps = pd.read_csv('dataset/timestamps_train.csv', usecols=['File', 'Start', 'End'])
    X_test_timestamps = pd.read_csv('dataset/timestamps_test.csv', usecols=['File', 'Start', 'End'])
    X_eval_timestamps = pd.read_csv('dataset/timestamps_eval.csv', usecols=['File', 'Start', 'End'])

    count = 0
    for file, start, end in X_train_timestamps:
        file_path = os.path.join(lared_dir, file)
        y, s = librosa.load(file_path, sr=sample_rate)
        cut_audio = y[start:end]
        output_file = str(count) + '.wav'
        output_file_path = os.path.join(dataset_dir, output_file)
        librosa.output.write_wav(output_file_path, cut_audio, sample_rate)
        count = count + 1
