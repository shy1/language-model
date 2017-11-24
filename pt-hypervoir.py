# pytorch implementation of:
# "Reservoir Computing on the Hypersphere" - M. Andrecut arXiv:1706.07896
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import re
import pickle
import time
from random import shuffle
from pytorch_tools import torchfold
# Reservoir computing on the hypersphere
import numpy as np
import chargrams as cg
from gensim.models.keyedvectors import KeyedVectors

wv = KeyedVectors.load_word2vec_format('/home/user01/dev/wang2vec/embeddings-i3e4-ssg-neg15-s1024w6.txt', binary=False)
temp = wv.index2word
glist = np.array(temp[1:len(temp)])
glist = [re.sub(r'_', ' ', j) for j in glist]
gramindex = {gram:idx for idx, gram in enumerate(glist)}

dtype = torch.cuda.FloatTensor
n = 2
stride = 1
leak = 0.382
step = 0
lrate = 0.002
bs = 1
trainsize = 64
i_size = 1024
h_size = 3072
torch.cuda.manual_seed(481639)

# V = torch.eye(i_size).type(torch.LongTensor)
U1 = torch.randn(h_size, i_size).type(torch.FloatTensor)
W1 = torch.zeros(i_size, h_size).type(dtype)

# normalize input matrix U to have zero mean and unit variance
def normalize_u(u):
    n_columns = u.size(1)
    for col in range(n_columns):
        u[:, col] = u[:, col] - u[:, col].mean()
        u[:, col] = u[:, col] / u[:, col].norm()
    return u

def indexToTensor(index):
    tensor = torch.zeros(1, i_size).type(dtype)
    tensor[0][index] = 1
    return tensor

def chunkToTensor(chunk):
    tensor = torch.zeros(len(chunk), 1, i_size).type(dtype)
    for pos, index in enumerate(chunk):
        tensor[pos][0][index] = 1
    return tensor

class Hypervoir(nn.Module):
    def __init__(self, i_size, h_size, leak):
        super(Hypervoir, self).__init__()
        self.leak = leak
        self.h_size = h_size
        self.lastidx = h_size - 1
        self.rollend = torch.FloatTensor(1, 1).type(dtype)
        self.rollfront = torch.FloatTensor(1, self.lastidx).type(dtype)

        self.hidden2out = nn.Linear(h_size, i_size, bias=False)
        self.softmax = nn.LogSoftmax()

    def init_hidden(self, bs):
        return Variable(torch.zeros(bs, self.h_size).type(dtype))

    def roll(self, hidden):
        # self.rollend[0] = hidden[0][self.lastidx]
        self.rollend = hidden.narrow(1, self.lastidx, 1)
        self.rollfront = hidden.narrow(1, 0, self.lastidx)
        hidden = torch.cat((self.rollend, self.rollfront), 1)
        return hidden
    # input (u_col) is the column from random input weight matrix U
    # with the same index as the input bigram
    def forward(self, u_col, hidden):
        hidden = (1.0 - self.leak) * hidden + self.leak * (u_col + self.roll(hidden))
        hidden = hidden / hidden.norm()
        output = self.hidden2out(hidden)
        output = self.softmax(output)
        return output, hidden

def rollCpu(hidd):
    lastidx = h_size - 1
    rollend = hidd.narrow(1, lastidx, 1)
    rollfront = hidd.narrow(1, 0, lastidx)
    hidd = torch.cat((rollend, rollfront), 1)
    return hidd

def nextHidden(hidd, ingot, leak):
    hidd = (1.0 - leak) * hidd + leak * (ingot + rollCpu(hidd))
    hidd = hidd / hidd.norm()

def createBatch(chunk, start, bsize, hidd, w_in, leak):
    # inputs = np.empty(length, dtype=np.int16)
    # ingots = torch.FloatTensor(b_size, i_size)
    batch = dict()
    targets = []
    states = torch.FloatTensor(b_size, h_size)
    for i in range(start, start + length):
        targets.append(chunk[i+1])
        ingot = w_in[:, chunk[i]]
        hidd = nextHidden(hidd, ingot, leak)
        states[i] = hidd
    batch["states"] = states
    batch["targets"] = targets

chunkfile = '/home/user01/dev/language-model/chunks256.p'
chunklist = pickle.load(open(chunkfile, "rb"))

trainchunks = []
for j in range(trainsize):
    chunk = chunklist[j]
    sgi = []
    for idx in range(0, len(chunk) - (n - 1), stride):
        try:
            sgi.append(gramindex[chunk[idx:idx + n]])
        except:
            print(chunk[idx:idx + n])
    intchunk = torch.cuda.LongTensor(sgi)
    trainchunks.append(intchunk)

U1 = normalize_u(U1)
model = Hypervoir(i_size, h_size, leak)
model = model.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lrate)

for ep in range(64):
    shuffle(trainchunks)
    startp = time.perf_counter()
    eperr = 0
    eploss = 0
    count = 0
    for chunk in trainchunks:

        t = 0
        tm1 = len(chunk) - 1
        hidden = model.init_hidden(bs)
        chunkloss = 0
        chunkerr = 0
        model.zero_grad()
        for timestep in range(tm1):
            count += 1
            incolumn = Variable(U1[:, chunk[t]]).unsqueeze(0)
            target = Variable(torch.cuda.LongTensor([chunk[t+1]]))
            output, hidden = model(incolumn, hidden)
            # probs = output.data[0][target.data[0]]
            top_n, top_i = output.data.topk(1)
            eperr += top_i[0][0] != chunk[t+1]
            loss = criterion(output, target)
            # chunkloss += loss.data[0]
            eploss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t += 1

    eperr = (eperr / count) * 100
    eploss = (eploss / count) * 100
    elapsedp = time.perf_counter() - startp
    tm, ts = divmod(elapsedp, 60)
    print(ep, eperr, loss.data[0], eploss, "{}m {}s".format(int(tm), int(ts)))
