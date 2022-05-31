import argparse
import os
from os import listdir
from os.path import isfile

import librosa
import pandas as pd
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument('--sample_rate', type=str, default='8000')
parser.add_argument('--dataset_dir', type=str, required=True)

args = parser.parse_args()
sample_rate = int(args.sample_rate)
dataset_dir = args.dataset_dir

lared_dir = "lared"
lared_ds_dir = "lared_ds"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("--- Downsampling audio files to 8000Hz to match sample rate of model ---")
for audio_file in listdir(lared_dir):
    print(audio_file)

    audio_path = os.path.join(lared_dir, audio_file)
    if not isfile(audio_path):
        raise Exception(f"Not a file: {audio_file}")

    if not os.path.exists(audio_path):
        y, s = librosa.load(audio_path, sr=sample_rate)
        output_path = os.path.join(lared_ds_dir, audio_file)
        sf.write(output_path, y, s)

### Create audio data and store it in dataset directory
print(f'--- Creating fine-tuning dataset for {sample_rate}Hz ---')
audio_timestamps = pd.read_csv('fine_tuning_timestamps.csv', usecols=['File', 'Start', 'End'])
audio_id = 0
print('Loading new file into memory...')
prev_file_path = 'lared_ds\master_chan_01.wav'
y, s = librosa.load(prev_file_path, sr=sample_rate)
for index, row in audio_timestamps.iterrows():
    print(str(audio_id))
    if not isfile(os.path.join(dataset_dir, str(audio_id) + '.wav')):
        file_path = os.path.join(lared_ds_dir, row['File'] + '.wav')
        if not file_path == prev_file_path:
            print('Loading new file into memory...')
            y, s = librosa.load(prev_file_path, sr=sample_rate)
        margin = 0.5
        cut_audio = y[librosa.time_to_samples(float(row['Start'])-margin, s):librosa.time_to_samples(float(row['End'])+margin, s)]
        librosa.util.normalize(cut_audio)
        output_file = str(audio_id) + '.wav'
        output_file_path = os.path.join(dataset_dir, output_file)
        sf.write(output_file_path, cut_audio, sample_rate)
        prev_file_path = file_path
    else:
        print('File already created')
    audio_id = audio_id + 1
