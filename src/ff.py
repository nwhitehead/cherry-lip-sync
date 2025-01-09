
# Lip shapes from:
# https://graphicmama.com/blog/free-mouth-shapes-character-animator-puppet/

import os
import argparse
import random
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import torch.nn as nn
import torch
from torchinfo import summary
import wandb

from model import NeuralNet
from data import LipsyncDataset, AudioMFCC, Upsample, PadVisemes, RandomChunk
from util import log_loss_color, log_epoch_color, log_validation_color

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sorted
viseme_labels = ['Ah', 'D', 'Ee', 'F', 'L', 'M', 'Neutral', 'Oh', 'R', 'S', 'Uh', 'Woo']

# Audiorate
rate = 16000

# Hyper-parameters
feature_dims = 28
lookahead_frames = 6
input_size = feature_dims
hidden_size = 200
num_classes = len(viseme_labels)
num_epochs = 200
batch_size = 10
learning_rate = 0.001
batch_time = 200
validate_every = 1
seed = 1

model = NeuralNet(input_size, hidden_size, num_classes)
checkpoint_name = 'model'

wandb.init(
    project='LipSync',
    config={
        'architecture': 'LSTM 1-layer',
        'learning_rate': learning_rate,
        'hidden_size': hidden_size,
        'checkpoint_name': checkpoint_name,
        'epochs': num_epochs,
        'batch_time': batch_time,
        'lookahead_frames': lookahead_frames,
        'seed': seed,
    },
)

# Show model summary
summary(model, input_size=(1, input_size))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
transform = nn.Sequential(
    Upsample(),
    AudioMFCC(),
    PadVisemes(),
    RandomChunk(size=batch_time, seed=seed),
)
dataset = LipsyncDataset('./data/lipsync.parquet', transform=transform)

rng = torch.Generator().manual_seed(seed)
train_dataset, test_dataset, _ = torch.utils.data.random_split(dataset, [0.80, 0.20, 0.0], generator=rng)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

with logging_redirect_tqdm():

    for epoch in tqdm(range(num_epochs), total=num_epochs, desc='Epoch', colour='#FF80D0'):

        model.train()
        train_losses = 0.0

        for i, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc='Sample', leave=False, colour='#00D0FF'):
            # Move tensors to the configured device
            #print(epoch, i, sample['audio'].shape, sample['visemes'].shape)
            # audio is N C T -> float
            audio = sample['audio'].to(device)
            # visemes is N T -> float representing viseme
            visemes = sample['visemes'].to(device)

            # Input to model needs to be N T C
            x = audio.permute(0, 2, 1)
            outputs, _hn = model(x)
            # Outpus is now N T C where C is number of visemes, numbers are raw logits (no softmax or anything)
            # CrossEntropyLoss takes in N C T.
            # Now use lookahead to define relation between input timing and output expectations
            # Ignore first few predictions from model
            left = outputs[:, lookahead_frames:, :].permute(0, 2, 1)
            right = visemes[:, :-lookahead_frames]
            loss = criterion(left, right)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()
            train_losses += loss
            log_loss_color('Loss: ', f'{loss:.5f}')

        epoch_loss = train_losses / len(train_loader)
        log_epoch_color('Epoch loss: ', f'{epoch_loss:.5f}')
        wandb.log({'loss': epoch_loss})

        if (epoch + 1) % validate_every == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0

                # Validation step
                for i, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Sample', leave=False, colour='#FFD0FF'):
                    audio = sample['audio'].to(device)
                    visemes = sample['visemes'].to(device)

                    x = audio.permute(0, 2, 1)
                    outputs, _hn = model(x)
                    # outputs is N T C
                    _, predicted = torch.max(outputs.data, 2)
                    # predicted is N T -> viseme
                    left = predicted[:, lookahead_frames:]
                    right = visemes[:, :-lookahead_frames]
                    total += right.nelement()
                    correct += (left == right).sum().item()

                accuracy = 100 * correct / total
                wandb.log({'acc': accuracy})
                log_validation_color('Accuracy: ', f'{accuracy:.5f}%')
                torch.save(model.state_dict(), f'checkpoints/{checkpoint_name}-{epoch}.pt')
