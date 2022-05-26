import argparse
import os
import time

import torch
import configs
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch_utils
from torch import optim, nn
from tqdm import tqdm
from tensorboardX import SummaryWriter

fine_tuning_timestamps = 'fine_tuning_timestamps.csv'
checkpoint_path = './checkpoints/in_use/resnet_with_augmentation/best.pth.tar'
fine_tuning_checkpoints_dir = 'checkpoints_fine_tuning'

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='resnet_with_augmentation')
parser.add_argument('--dropout_rate', type=str, default='0.5')
parser.add_argument('--sample_rate', type=str, default='8000')
parser.add_argument('--dataset_dir', type=str, required=True)
parser.add_argument('--gradient_accumulation_steps', type=str, default='1')

args = parser.parse_args()
config = configs.CONFIG_MAP[args.config]
dataset_dir = args.dataset_dir
dropout_rate = float(args.dropout_rate)
sample_rate = int(args.sample_rate)
gradient_accumulation_steps = int(args.gradient_accumulation_steps)
learning_rate = 0.01  # Learning rate.
decay_rate = 0.9999  # Learning rate decay per minibatch.
min_learning_rate = 0.000001  # Minimum learning rate.


### Class representing the LaRed dataset
class LaRedDataset(Dataset):
    def __init__(self, audio_labels, audio_dir):
        self.audio_labels = audio_labels
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.audio_dir)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, str(idx) + '.wav')
        audio = librosa.load(audio_path, sr=sample_rate)
        label = self.audio_labels[idx]
        return audio, label


### Load dataset and split into training and validation sets
print('--- Loading dataset and split into training and validation sets ---')
labels = pd.read_csv(fine_tuning_timestamps, usecols=['Label'])
dataset = LaRedDataset(labels, dataset_dir)
batch_size = 64
validation_split = 0.2
shuffle_dataset = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

### Load model and checkpoints
print('--- Loading model and checkpoints ---')
device = torch.device('cpu')
model = config['model'](dropout_rate=dropout_rate, linear_layer_size=config['linear_layer_size'],
                        filter_sizes=config['filter_sizes'])
model.set_device(device)
optimizer = optim.Adam(model.parameters())
torch_utils.load_checkpoint(checkpoint_path, model, optimizer)
writer = SummaryWriter(fine_tuning_checkpoints_dir)


# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['state_dict'])

def run_training_loop(n_epochs, model, device, checkpoint_dir,
                      optimizer, iterator, log_frequency=25, val_iterator=None, gradient_clip=1.,
                      verbose=True):
    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = run_epoch(model, 'train', device, iterator,
                               checkpoint_dir=checkpoint_dir, optimizer=optimizer,
                               log_frequency=log_frequency, checkpoint_frequency=log_frequency,
                               clip=gradient_clip, val_iterator=val_iterator,
                               verbose=verbose)

        if verbose:
            end_time = time.time()
            epoch_mins, epoch_secs = torch_utils.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')


def run_epoch(model, mode, device, iterator, checkpoint_dir, optimizer=None, clip=None,
              batches=None, log_frequency=None, checkpoint_frequency=None,
              validate_online=True, val_iterator=None, val_batches=None,
              verbose=True):
    """ args:
            mode: 'train' or 'eval'
    """

    def _eval_for_logging(model, device, val_itr, val_iterator, val_batches_per_log):
        model.eval()
        val_losses = [];
        val_accs = []

        for j in range(val_batches_per_log):
            try:
                val_batch = val_itr.next()
            except StopIteration:
                val_itr = iter(val_iterator)
                val_batch = val_itr.next()

            val_loss, val_acc = _eval_batch(model, device, val_batch)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

        model.train()
        return val_itr, np.mean(val_losses), np.mean(val_accs)

    def _eval_batch(model, device, batch, batch_index=None, clip=None):
        if batch is None:
            print("None Batch")
            return 0.

        with torch.no_grad():
            seqs, labs = batch

            src = torch.from_numpy(np.array(seqs)).float().to(device)
            trg = torch.from_numpy(np.array(labs)).float().to(device)
            output = model(src).squeeze()

            criterion = nn.BCELoss()
            bce_loss = criterion(output, trg)
            preds = torch.round(output)
            acc = torch.sum(preds == trg).float() / len(trg)  # sum(preds==trg).float()/len(preds)

            return bce_loss.item(), acc.item()

    def _train_batch(model, device, batch, batch_index=None, clip=None):

        if batch is None:
            print("None Batch")
            return 0.

        seqs, labs = batch

        src = torch.from_numpy(np.array(seqs)).float().to(device)
        trg = torch.from_numpy(np.array(labs)).float().to(device)

        # optimizer.zero_grad()

        output = model(src).squeeze()

        criterion = nn.BCELoss()

        preds = torch.round(output)
        acc = torch.sum(preds == trg).float() / len(trg)

        bce_loss = criterion(output, trg)

        loss = bce_loss
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if model.global_step % gradient_accumulation_steps == 0:
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            model.zero_grad()

        return bce_loss.item(), acc.item()

    if not (bool(iterator) ^ bool(batches)):
        raise Exception("Must pass either `iterator` or batches")

    if mode.lower() not in ['train', 'eval']:
        raise Exception("`mode` must be 'train' or 'eval'")

    if mode.lower() == 'train' and validate_online:
        val_batches_per_epoch = torch_utils.num_batches_per_epoch(val_iterator)
        val_batches_per_log = int(np.round(val_batches_per_epoch))

        val_itr = iter(val_iterator)

    if mode is 'train':
        if optimizer is None:
            raise Exception("Must pass Optimizer in train mode")
        model.train()
        _run_batch = _train_batch
    elif mode is 'eval':
        model.eval()
        _run_batch = _eval_batch

    epoch_loss = 0

    optimizer = optim.Adam(model.parameters())

    if iterator is not None:
        batches_per_epoch = torch_utils.num_batches_per_epoch(iterator)
        batch_losses = [];
        batch_accs = [];
        batch_consistency_losses = [];
        batch_ent_losses = []

        for i, batch in tqdm(enumerate(iterator)):
            # learning rate scheduling
            lr = (learning_rate - min_learning_rate) * decay_rate ** (float(model.global_step)) + min_learning_rate
            optimizer.lr = lr

            batch_loss, batch_acc = _run_batch(model, device, batch,
                                               batch_index=i, clip=clip)

            batch_losses.append(batch_loss);
            batch_accs.append(batch_acc)

            if log_frequency is not None and (model.global_step + 1) % log_frequency == 0:
                val_itr, val_loss_at_step, val_acc_at_step = _eval_for_logging(model, device,
                                                                               val_itr, val_iterator,
                                                                               val_batches_per_log)

                is_best = (val_loss_at_step < model.best_val_loss)
                if is_best:
                    model.best_val_loss = val_loss_at_step

                train_loss_at_step = np.mean(batch_losses)
                train_acc_at_step = np.mean(batch_accs)

                if verbose:
                    print("\nLogging at step: ", model.global_step)
                    print("Train loss: ", train_loss_at_step)
                    print("Train accuracy: ", train_acc_at_step)
                    print("Val loss: ", val_loss_at_step)
                    print("Val accuracy: ", val_acc_at_step)

                writer.add_scalar('loss/train', train_loss_at_step, model.global_step)
                writer.add_scalar('acc/train', train_acc_at_step, model.global_step)
                writer.add_scalar('loss/eval', val_loss_at_step, model.global_step)
                writer.add_scalar('acc/eval', val_acc_at_step, model.global_step)
                batch_losses = [];
                batch_accs = []  # reset

            if checkpoint_frequency is not None and (model.global_step + 1) % checkpoint_frequency == 0:
                state = torch_utils.make_state_dict(model, optimizer, model.epoch,
                                                    model.global_step, model.best_val_loss)
                torch_utils.save_checkpoint(state, is_best=is_best, checkpoint=checkpoint_dir)

            epoch_loss += batch_loss
            model.global_step += 1

        model.epoch += 1
        return epoch_loss / len(iterator)


print('--- Training the model ---')
run_training_loop(n_epochs=1, model=model, device=device,
                  iterator=train_loader, checkpoint_dir=fine_tuning_checkpoints_dir, optimizer=optimizer,
                  val_iterator=validation_loader,
                  verbose=True)
