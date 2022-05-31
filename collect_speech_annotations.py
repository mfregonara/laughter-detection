import os
from os import listdir

import librosa
import pandas as pd
import soundfile as sf

sample_rate = 8000
lared_dir = "lared"
lared_ds_dir = "lared_ds"

# print("--- Downsampling audio files to 8000Hz to match sample rate of model ---")
# for audio_file in listdir(lared_dir):
#     print(audio_file)
#
#     audio_path = os.path.join(lared_dir, audio_file)
#     if not isfile(audio_path):
#         raise Exception(f"Not a file: {audio_file}")
#
#     if not os.path.exists(audio_path):
#         y, s = librosa.load(audio_path, sr=sample_rate)
#         output_path = os.path.join(lared_ds_dir, audio_file)
#         sf.write(output_path, y, s)

total_instances = {}
for audio_file in listdir(lared_ds_dir):
    audio_path = os.path.join(lared_ds_dir, audio_file)
    y, s = librosa.load(audio_path, sr=sample_rate)
    instances = librosa.split(y, top_db=60)
    total_instances[str(audio_file)] = instances

speech_annotations = pd.DataFrame(columns=['File', 'Start', 'End', 'Label'])
laughter_annotations = pd.read_csv('fine_tuning_timestamps.csv', header=0, index=None)

for audio_file, speech_timestamps in total_instances.items():
    for speech in speech_timestamps:
        start = librosa.samples_to_time(speech[0], sample_rate)
        end = librosa.samples_to_time(speech[1], sample_rate)

        is_laughter_flag = False
        laughter_annotations_file = laughter_annotations.loc[laughter_annotations['File'] == audio_file]
        for i, r in laughter_annotations_file.iterrows():
            if r['Start'] < start < r['End'] or r['Start'] < end < r['End']:
                is_laughter_flag = True

        if not is_laughter_flag:
            speech_annotations.loc[len(speech_annotations.index)] = [audio_file, start, end, 0]

speech_annotations = speech_annotations.sample(802)
laughter_annotations.append(speech_annotations)
laughter_annotations.to_csv('fine_tuning_timestamps.csv')

print("Results saved in csv file.")
