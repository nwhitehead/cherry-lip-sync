
# Lip shapes from:
# https://graphicmama.com/blog/free-mouth-shapes-character-animator-puppet/

import os
import argparse
import logging
import random
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import torch.nn as nn
import torch
from torchinfo import summary

from model import NeuralNet
from data import LipsyncDataset, AudioMFCC, Upsample, PadVisemes, RandomChunk

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
validate_every = 5

model = NeuralNet(input_size, hidden_size, num_classes)

summary(model, input_size=(1, input_size))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
transform = nn.Sequential(
    Upsample(),
    AudioMFCC(),
    PadVisemes(),
    RandomChunk(size=batch_time, seed=1),
)
dataset = LipsyncDataset('./data/lipsync.parquet', transform=transform)

rng = torch.Generator().manual_seed(1)
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

LOG = logging.getLogger('Training')
logging.basicConfig(level=logging.INFO)

def hex_to_rgb(hex_color):
    """Converts a hex color code (6 digit) to RGB integers."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def truecolor(msg, color):
    r, g, b = hex_to_rgb(color)
    return f'\x1b[38;2;{r};{g};{b}m{msg}\x1b[0m'

def log_loss_color(prefix, msg):
    LOG.info(f'{prefix}{truecolor(msg, "#80ff80")}')
def log_epoch_color(prefix, msg):
    LOG.info(f'{prefix}{truecolor(msg, "#ffff80")}')
def log_validation_color(prefix, msg):
    LOG.info(f'{prefix}{truecolor(msg, "#fff0f0")}')

with logging_redirect_tqdm():

    for epoch in tqdm(range(num_epochs), total=num_epochs, desc='Epoch', colour='#FF80D0'):

        train_losses = 0.0

        for i, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc='Sample', leave=False, colour='#00D0FF'):
            # Move tensors to the configured device
            #print(epoch, i, sample['audio'].shape, sample['visemes'].shape)
            # audio is N C T -> float
            audio = sample['audio'].to(torch.float).to(device)
            # visemes is N T -> float representing viseme
            visemes = sample['visemes'].to(torch.long).to(device)

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

        log_epoch_color('Epoch loss: ', f'{(train_losses / len(train_loader)):.5f}')

        if (epoch + 1) % validate_every == 0:
            with torch.no_grad():
                correct = 0
                total = 0

                # Validation step
                for i, sample in tqdm(enumerate(test_loader), total=len(test_loader), desc='Sample', leave=False, colour='#FFD0FF'):
                    audio = sample['audio'].to(torch.float).to(device)
                    visemes = sample['visemes'].to(torch.long).to(device)

                    x = audio.permute(0, 2, 1)
                    outputs, _hn = model(x)
                    # outputs is N T C
                    _, predicted = torch.max(outputs.data, 2)
                    # predicted is N T -> viseme
                    left = predicted[:, lookahead_frames:]
                    right = visemes[:, :-lookahead_frames]
                    total += right.nelement()
                    correct += (left == right).sum().item()

                log_validation_color('Accuracy: ', f'{(100 * correct / total):.5f}%')
