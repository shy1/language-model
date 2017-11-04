from __future__ import unicode_literals, print_function, division
from io import open
import glob

def findFiles(path): return glob.glob(path)

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
low_letters = string.ascii_lowercase + " "
n_letters = len(low_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

category_lines = {}
all_categories = []

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def makeLower(upper):
    return str.lower(upper)

# change findFiles parameter to appropriate relative path
for filename in findFiles('/home/user01/dev/data/gutenberg/clean/dickens/*.txt'):
    temp = ''
    category = filename.split('/')[-1].split('.')[0]
    #category = category.lower()
    #print(category)
    all_categories.append(category)
    lines = readLines(filename)

    for line in lines:
        temp = temp + '\n' + line

    #remove first \n at beginning of string/file
    temp = temp[1:]

    # write utf-8 encoded data to new file and read back in as standard python string
    # more expedient then learning intricacies of python2 -> python3 unicode/ascii conversion
    temp1 = category + '1.txt'
    with open(temp1, 'w') as newfile:
        newfile.write(temp)
    with open(temp1, 'r') as newread:
        lines = newread.read().lower().split('\n')
        lines2 = [' {0} '.format(line) for line in lines]
    category_lines[category] = lines2
n_categories = len(all_categories)


import torch


def letterToIndex(letter):
    return low_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    # use multiple characters as input, currently 2 (n_letters * 2)
    tensor = torch.zeros(len(line) - 1, 1, n_letters * 2).cuda()
    #print(len(line))

    # note: check how python for loops handle ranges (inclusive/exclusive/etc)
    for idx in range(0, len(line) - 1):
        index1 = letterToIndex(line[idx])
        index2 = letterToIndex(line[idx + 1]) + 27
        output = str(idx) + " " + str(index1) + " " + str(index2)
        tensor[idx][0][index1] = 1
        tensor[idx][0][index2] = 1
        #print(line[idx])
        #print(line[idx + 1])
        #print(output)
    return tensor



import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size).cuda())

n_hidden = 486
rnn = RNN(n_letters * 2, n_hidden, n_categories)
rnn.cuda()




def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return all_categories[category_i], category_i


import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.cuda.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line).cuda())
    return category, line, category_tensor, line_tensor


criterion = nn.NLLLoss()

learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor, hidden)

    loss = criterion(output, category_tensor)
    loss.backward()


    optimizer.step()

    return output, loss.data[0]


import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000


current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1


for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()
