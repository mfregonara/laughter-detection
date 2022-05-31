import librosa
import numpy as np
import os
import sys
import torch
import jpype
jpype.startJVM()
import asposecells
from asposecells.api import Workbook, FileFormatType
from os import listdir
from os.path import isfile, join
import laugh_segmenter
from tqdm import tqdm
from functools import partial

import audio_utils
import data_loaders
import tgt
import torch_utils

sys.path.append('./utils/')


def predict(config, device, model_path, audio_dir, sample_rate, threshold,
            min_length, output_dir, lared_sample_rate):
    # Load the Model
    print("--- Loading the model ---")
    model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'],
                            filter_sizes=config['filter_sizes'])
    feature_fn = config['feature_fn']
    model.set_device(device)

    if os.path.exists(model_path):
        torch_utils.load_checkpoint(model_path + '/best.pth.tar', model)
        model.eval()
    else:
        raise Exception(f"Model checkpoint not found at {model_path}")

    # Iterate over audio files
    print("--- Making predictions ---")
    total_instances = {}
    number_laughs = 0
    for audio_file in listdir(audio_dir):
        audio_path = join(audio_dir, audio_file)
        print(audio_file)
        if not isfile(audio_path):
            raise Exception(f"Not a file: {audio_file}")

        # Load the audio file and features
        inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
            audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate)

        collate_fn = partial(audio_utils.pad_sequences_with_labels,
                             expand_channel_dim=config['expand_channel_dim'])

        inference_generator = torch.utils.data.DataLoader(
            inference_dataset, num_workers=0, batch_size=8, shuffle=False, collate_fn=collate_fn)

        # Make Predictions
        probs = []
        for model_inputs, _ in tqdm(inference_generator):
            x = torch.from_numpy(model_inputs).float().to(device)
            preds = model(x).cpu().detach().numpy().squeeze()
            if len(preds.shape) == 0:
                preds = [float(preds)]
            else:
                preds = list(preds)
            probs += preds
        probs = np.array(probs)

        file_length = audio_utils.get_audio_length(audio_path)

        fps = len(probs) / float(file_length)

        probs = laugh_segmenter.lowpass(probs)
        instances = laugh_segmenter.get_laughter_instances(probs, threshold=threshold, min_length=float(min_length),
                                                           fps=fps)
        print("found %d laughs." % (len(instances)))
        total_instances[str(audio_file)] = instances
        number_laughs = number_laughs + len(instances)

        # # Save to textgrid
        # if len(instances) > 0:
        #     laughs = [{'start': i[0], 'end': i[1]} for i in instances]
        #     tg = tgt.TextGrid()
        #     laughs_tier = tgt.IntervalTier(name='laughter', objects=[
        #         tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs])
        #     tg.add_tier(laughs_tier)
        #     text_dir = os.path.join(output_dir, str(lared_sample_rate) + 'Hz')
        #     fname = os.path.splitext(audio_file)[0]
        #     if not os.path.exists(text_dir):
        #         os.makedirs(text_dir)
        #     tgt.write_to_file(tg, os.path.join(text_dir, fname + '_laughter.TextGrid'))
        #
        #     print('Saved laughter segments in {}'.format(
        #         os.path.join(text_dir, fname + '_laughter.TextGrid')))

    # Save to workbook
    workbook = Workbook(FileFormatType.XLSX)
    sheet = workbook.getWorksheets().get(0)
    cells = sheet.getCells()

    cells.get(0, 0).putValue("File")
    cells.get(0, 1).putValue("Start")
    cells.get(0, 2).putValue("End")
    cells.get(0, 3).putValue("Duration")

    row = 1
    for audio_file, laughter_timestamps in total_instances.items():
        parent_name = os.path.splitext(audio_file)[0][:13]
        segment_number = os.path.splitext(audio_file)[0][14:]
        segment_duration = 9900
        previous_duration = segment_duration * (segment_number-1)
        for laughter in laughter_timestamps:
            cells.get(row, 0).putValue(parent_name)
            cells.get(row, 1).putValue(round(previous_duration + laughter[0], 2))
            cells.get(row, 2).putValue(round(previous_duration + laughter[1], 2))
            cells.get(row, 3).putValue(round(float(laughter[1] - laughter[0]), 2))
            row = row + 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    workbook_dir = os.path.join(output_dir, str(lared_sample_rate) + 'Hz')
    workbook.save(workbook_dir + ".csv")
    jpype.shutdownJVM()
    print("Results saved in workbook.")

    print();
    print("Found a total of %d laughs." % (number_laughs))
