import librosa
import numpy as np
import os
import sys
import torch
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
    total_instances = []
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
        total_instances = total_instances + instances

        # Save to textgrid
        if len(instances) > 0:
            laughs = [{'start': i[0], 'end': i[1]} for i in instances]
            tg = tgt.TextGrid()
            laughs_tier = tgt.IntervalTier(name='laughter', objects=[
                tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs])
            tg.add_tier(laughs_tier)
            text_dir = os.path.join(output_dir, str(lared_sample_rate) + 'Hz')
            fname = os.path.splitext(audio_file)[0]
            if not os.path.exists(text_dir):
                os.makedirs(text_dir)
            tgt.write_to_file(tg, os.path.join(text_dir, fname + '_laughter.TextGrid'))

            print('Saved laughter segments in {}'.format(
                os.path.join(text_dir, fname + '_laughter.TextGrid')))

    print();
    print("Found a total of %d laughs." % (len(total_instances)))


