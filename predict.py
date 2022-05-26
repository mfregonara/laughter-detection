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
import soundfile as sf
from scipy.signal import butter, filtfilt

from segment_laughter import predict

sys.path.append('./utils/')

sample_rate = 8000
lared_dir = "lared"
lared_ds_dir = "lared_ds"
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

print("--- Downsampling audio files to 8000Hz to match sample rate of model ---")
for audio_file in listdir(lared_dir):
    print(audio_file)

    audio_path = os.path.join(lared_dir, audio_file)
    if not isfile(audio_path):
        raise Exception(f"Not a file: {audio_file}")

    y, s = librosa.load(audio_path, sr=sample_rate)
    output_path = os.path.join(lared_ds_dir, audio_file)
    sf.write(output_path, y, s)

cutoff_f = 0.5 * lared_sample_rate
print(f"--- Lowpassing audio files to {cutoff_f}Hz to simulate downsampling to {lared_sample_rate}Hz ---")
for audio_file in listdir(lared_ds_dir):
    print(audio_file)

    audio_path = os.path.join(lared_ds_dir, audio_file)
    if not isfile(audio_path):
        raise Exception(f"Not a file: {audio_file}")

    y, s = librosa.load(audio_path, sr=sample_rate)
    w = cutoff_f / (sample_rate / 2)
    b, a = butter(5, w, 'low')
    output = filtfilt(b, a, y)

    output_path = os.path.join(lared_lp_dir, audio_file)
    sf.write(output_path, output, s)


# predict(config, device, model_path, lared_lp_dir, sample_rate, threshold,
#         min_length, output_dir, lared_sample_rate)

