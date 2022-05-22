import argparse
import librosa
import os
import sys
import torch
from os import listdir
from os.path import isfile
import configs
import soundfile as sf

from segment_laughter import predict

sys.path.append('./utils/')

sample_rate = 8000
lared_dir = "lared"
lared_ds_dir = "lared_ds"

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

print(f"--- Downsampling audio files to {lared_sample_rate}Hz ---")

for audio_file in listdir(lared_dir):
    print(audio_file)

    audio_path = os.path.join(lared_dir, audio_file)
    if not isfile(audio_path):
        raise Exception(f"Not a file: {audio_file}")

    y, s = librosa.load(audio_path, sr=lared_sample_rate)
    output_path = os.path.join(lared_ds_dir, audio_file)
    sf.write(output_path, y, s)

predict(config, device, model_path, lared_ds_dir, sample_rate, threshold,
        min_length, output_dir, lared_sample_rate)

