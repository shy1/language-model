# https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import string
import glob

from tensorboardX import SummaryWriter
from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default=1200)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=486)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.00062)
argparser.add_argument('--chunk_len', type=int, default=243)
args = argparser.parse_args()

writer = SummaryWriter('runs/test3')

all_characters = string.ascii_lowercase + string.digits + string.punctuation + " "
n_characters = len(all_characters)

def findFiles(path): return glob.glob(path)
filelist = []

for filename in findFiles(args.filename):
    file, file_len = read_file(filename)
    filelist.append(file)

print(len(filelist))
def random_training_set(chunk_len, file, file_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk = file[start_index:end_index]
    inp = char_tensor(chunk[:-1]).cuda()
    target = char_tensor(chunk[1:]).cuda()
    return inp, target

decoder = RNN(n_characters, args.hidden_size, n_characters, args.n_layers)
decoder.cuda()
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0



def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len

def save():
    temp0 = os.path.split(args.filename)
    temp1 = os.path.split(temp0[0])
    dir2 = temp1[1]
    dir1 = os.path.split(temp1[0])[1]
    save_filename = dir1 + "-" + dir2 + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

try:
    print("Training for %d epochs..." % args.n_epochs)
    for file in filelist:
        file_len = len(file)
        for epoch in range(1, args.n_epochs + 1):
            loss = train(*random_training_set(args.chunk_len, file, file_len))
            writer.add_scalar('data/loss', loss, epoch)
            loss_avg += loss

            if epoch % args.print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
                print(generate(decoder, 'wh', 10), '\n')

        #import matplotlib.pyplot as plt
        #import matplotlib.ticker as ticker

        #plt.figure()
        #plt.plot(all_losses)

    writer.close()
    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    writer.close()
    save()
