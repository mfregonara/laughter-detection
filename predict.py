# python3.6 predict.py --lared_sample_rate=300 --output_dir=output_no_fine_tuning --min_length=0.2 --threshold=0.5 --model_path=checkpoints/in_use/resnet_with_augmentation
# 300, 350, 500, 800, 1250, 2000, 3150, 5000, 8000, 12000, 20000, 30000, 44100


import argparse
import librosa
import os
import sys

import torch
from os import listdir
from os.path import isfile
import configs
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from pydub import AudioSegment

from segment_laughter import predict

sys.path.append('./utils/')

sample_rate = 44100
lared_dir = "lared"
lared_split_dir = "lared_split"
lared_lp_dir = "lared_lp"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='checkpoints/in_use/resnet_with_augmentation')
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--threshold', type=str, default='0.5')
parser.add_argument('--min_length', type=str, default='0.2')
parser.add_argument('--lared_sample_rate', type=str, default='44100')
parser.add_argument('--output_dir', type=str, default=None)
args = parser.parse_args()

model_path = args.model_path
config = configs.CONFIG_MAP[args.config]
threshold = float(args.threshold)
min_length = float(args.min_length)
lared_sample_rate = int(args.lared_sample_rate)
output_dir = args.output_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

for audio_file in listdir(lared_dir):
    data, sr = librosa.load(os.path.join(lared_dir, audio_file), sample_rate)
    librosa.util.normalize(data)
    sf.write(os.path.join(lared_dir, audio_file), data, sr)

for audio_file in listdir(lared_dir):
    print(audio_file)
    full_audio = AudioSegment.from_wav(os.path.join(lared_dir, audio_file))
    split_duration = 1000 * (full_audio.duration_seconds / 100)
    for i in range(0, 100):
        split_audio = full_audio[i*split_duration:(i+1)*split_duration]
        split_audio.export(os.path.join(lared_split_dir, os.path.splitext(audio_file)[0]) + "_" + str(i+1) + ".wav", format="wav")


def butter_lowpass_filter(data, cutoff, sample_rate, order):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


cutoff_f = 0.5 * lared_sample_rate
if cutoff_f <= 20000:
    print(f"--- Lowpassing audio files to {cutoff_f}Hz to simulate downsampling to {lared_sample_rate}Hz ---")
    for audio_file in listdir(lared_split_dir):
        print(audio_file)

        audio_path = os.path.join(lared_split_dir, audio_file)
        if not isfile(audio_path):
            raise Exception(f"Not a file: {audio_file}")
        sr, data = wavfile.read(audio_path)

        filtered = butter_lowpass_filter(data, cutoff_f, sample_rate, 6)

        output_path = os.path.join(lared_lp_dir, audio_file)
        wavfile.write(output_path, sr, filtered)

predict(config, device, model_path, lared_lp_dir, sample_rate, threshold,
        min_length, output_dir, lared_sample_rate)
