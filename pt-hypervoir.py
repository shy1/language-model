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
lrate = 0.0004
bs = 192
trainsize = 64
i_size = 1024
h_size = 6144
torch.cuda.manual_seed(481639)

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
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, bs):
        return Variable(torch.zeros(bs, self.h_size).type(dtype), requires_grad=False)

    def roll(self, hidden):
        # self.rollend[0] = hidden[0][self.lastidx]
        self.rollend = hidden.narrow(1, self.lastidx, 1)
        self.rollfront = hidden.narrow(1, 0, self.lastidx)
        hidden = torch.cat((self.rollend, self.rollfront), 1)
        return hidden
    # input (u_col) is the column from random input weight matrix U
    # with the same index as the input bigram
    def forward(self, u_col, hidden, t):
        if t > 0:
            hidden = (1.0 - self.leak) * hidden + self.leak * (u_col + self.roll(hidden))
        else:
            # hidden = (1.0 - self.leak) * hidden + self.leak * (u_col + self.roll(hidden))
            hidden = u_col
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
    return hidd

def primeStates(chunk, w_in, leak):
    ## set first hidden state directly as the input weight matrix's column
    # rather than using leaky combinations with an inital all-zeros state
    ## in the iniitial zeros case the resulting values for the new state
    # would be scaled down by the leak rate without a corresponding increase
    # from the previous states contribution since (1 - leak) * zeros and
    # roll(zeros) both = 0
    hidd = w_in[:, chunk[0]].unsqueeze(0)
    hidd = hidd / hidd.norm()
    for i in range(1, len(chunk)):
        ingot = w_in[:, chunk[i]]
        hidd = nextHidden(hidd, ingot, leak)
    return hidd

def getStates(chunk, w_in, leak):
    processed = dict()
    primechunk = chunk[0:8]
    chunk = chunk[8:len(chunk)]
    hidd = primeStates(primechunk, w_in, leak)
    length = len(chunk) - 1
    targets = torch.LongTensor(length)
    states = torch.FloatTensor(length, h_size)
    ingots = torch.FloatTensor(length, h_size)
    for i in range(length):
        targets[i] = chunk[i+1]
        ingot = w_in[:, chunk[i]].unsqueeze(0)
        hidd = nextHidden(hidd, ingot, leak)
        states[i] = hidd
    processed["states"] = states
    processed["targets"] = targets
    return processed

def getSomeStates(chunk, start, bsize, hidd, w_in, leak):
    # inputs = np.empty(length, dtype=np.int16)
    # ingots = torch.FloatTensor(b_size, i_size)
    processed = dict()
    targets = []
    states = torch.FloatTensor(b_size, h_size)
    for i in range(start, start + length):
        targets.append(chunk[i+1])
        ingot = w_in[:, chunk[i]]
        hidd = nextHidden(hidd, ingot, leak)
        states[i] = hidd
    processed["states"] = states
    processed["targets"] = targets
    return processed

# V = torch.eye(i_size).type(torch.LongTensor)
U1 = torch.randn(h_size, i_size).type(dtype)
U1 = normalize_u(U1)
W1 = Variable(torch.zeros(h_size, i_size).type(dtype), requires_grad=True)

chunkfile = '/home/user01/dev/language-model/chunks256.p'
chunklist = pickle.load(open(chunkfile, "rb"))

trainchunks = []
for j in range(trainsize):
    chunk = chunklist[j]
    sgi = []
    for idx in range(0, bs + 9, stride):
        try:
            sgi.append(gramindex[chunk[idx:idx + n]])
        except:
            print(chunk[idx:idx + n])
    intchunk = torch.LongTensor(sgi)
    trainchunks.append(intchunk)

# model = Hypervoir(i_size, h_size, leak)
# model = model.cuda()
# criterion = nn.NLLLoss()
optimizer = optim.Adam([W1], lr=lrate)


states = torch.Tensor(len(trainchunks), bs, h_size).type(dtype)
targets = torch.Tensor(len(trainchunks), bs).type(torch.cuda.LongTensor)
for i in range(len(trainchunks)):
    batch = getStates(trainchunks[i], U1, leak)
    states[i] = batch["states"]
    targets[i] = batch["targets"]

cstates = Variable(states.cuda(), requires_grad=False)
ctargets = Variable(targets.cuda(), requires_grad=False)
startp = time.perf_counter()
for ep in range(4096):
    shuffle(trainchunks)

    # eperr = 0
    eploss = 0
    count = ep + 1
    for c in range(len(trainchunks)):
        chunkloss = 0
        chunkerr = 0
#        print(W1.size(), cstates[c].size())
        logits = cstates[c].mm(W1)
#        logits = W1.mm(cstates[c])
        output = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(output, ctargets[c])
        # chunkloss += loss.data[0]
        eploss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # eperr = (eperr / count) * 100
    eploss = (eploss / len(trainchunks))
    if count % 128 == 0:
        elapsedp = time.perf_counter() - startp
        tm, ts = divmod(elapsedp, 60)
        print(count, loss.data[0], eploss, "{}m {}s".format(int(tm), int(ts)))

# for ep in range(32):
#     # shuffle(trainchunks)
#     startp = time.perf_counter()
#     eperr = 0
#     eploss = 0
#     count = 0
#     for chunk in trainchunks:
#
#         t = 0
#         tm1 = len(chunk) - 1
#         hidden = model.init_hidden(bs)
#         chunkloss = 0
#         chunkerr = 0
#         # model.zero_grad()
#         for timestep in range(tm1):
#             count += 1
#             incolumn = Variable(U1[:, chunk[t]], requires_grad=False).unsqueeze(0)
#             target = Variable(torch.cuda.LongTensor([chunk[t+1]]), requires_grad=False)
#             output, hidden = model(incolumn, hidden, t)
#             # probs = output.data[0][target.data[0]]
#             top_n, top_i = output.data.topk(1)
#             eperr += top_i[0][0] != chunk[t+1]
#             loss = criterion(output, target)
#             # chunkloss += loss.data[0]
#             eploss += loss.data[0]
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             t += 1
#
#     eperr = (eperr / count) * 100
#     eploss = (eploss / count)
#     elapsedp = time.perf_counter() - startp
#     tm, ts = divmod(elapsedp, 60)
#     print(ep, eperr, loss.data[0], eploss, "{}m {}s".format(int(tm), ts))
