## Shallow Ensemble of Temporal Hypersphere Reservoirs
## - pre-trains a single reservoir for later inclusion in an ensemble

import numpy as np
import cupy as cp
import chargrams as cg
import re
import pickle
import time
from random import shuffle

from gensim.models.keyedvectors import KeyedVectors

import pydybm.arraymath as amath
import pydybm.arraymath.dycupy as dycupy
from pydybm.base.sgd32 import ADAM

# todo: grab bigram indexes directly from text file instead of loading w2vec library
wv = KeyedVectors.load_word2vec_format('/home/user01/dev/wang2vec/embeddings-i3e4-ssg-neg15-s1024w6.txt', binary=False)
temp = wv.index2word
glist = np.array(temp[1:len(temp)])
glist = [re.sub(r'_', ' ', j) for j in glist]
gramindex = {gram:idx for idx, gram in enumerate(glist)}

def init(M, N, inweights):
    v = cp.identity(M, dtype=np.float32)
    for key in inweights:
        for m in range(M):
            inweights[key][:, m] = inweights[key][:, m] - inweights[key][:, m].mean()
            inweights[key][:, m] = inweights[key][:, m] / cp.linalg.norm(inweights[key][:, m])
    return inweights, v

def train_kcpa(inweights, v, variables, leak, bs, step, s, cpstates):
    T = len(s)
    N = 1024
    M = 1024
    x1 = cp.zeros(N * layerscales["L1"], dtype=np.float32)
    # gradient = dict()
    # softerr1 = 0
    # err1 = 0
    skipfirst = 1
    t = step
    tm1 = (T - 1 - t - skipfirst)

    for k in range(skipfirst):
        current = s[t - step]
        x1 = (1.0 - leak) * x1 + leak * (inweights["U1"][:, current] + cp.roll(x1, 1))
        x1 = x1 / cp.linalg.norm(x1)
        # wx = cp.dot(variables["W1"], x1)
        # wx = wx - cp.max(wx)
        # p = cp.exp(wx)
        # p1 = p / cp.sum(p)
        t += 1

    for b1 in range(tm1):
        current = s[t - step]
        x1 = (1.0 - leak) * x1 + leak * (inweights["U1"][:, current] + cp.roll(x1, 1))
        x1 = x1 / cp.linalg.norm(x1)
        # wx = cp.dot(variables["W1"], x1)
        # wx = wx - cp.max(wx)
        # p = cp.exp(wx)
        # p1 = p / cp.sum(p)

        cpstates = cp.concatenate((cpstates, x1.reshape((1, N * layerscales["L1"]))))
        # target = s[t+1]

        # gradient["W1"] = cp.outer(v[:, target] - p1, x1)
        # SGD.update_state(gradient)
        # delta = SGD.get_delta()
        # SGD.update_with_L1_regularization(variables, delta, L1)
        t += 1

    return variables, cpstates

def train(inweights, v, variables, leak, bs, steps, testflag, s, count):
    T = len(s)
    N = 1024
    M = 1024
    x1 = cp.zeros(N * layerscales["L1"], dtype=np.float32)
    gradient = dict()
    softerr1 = 0
    err1 = 0
    skipfirst = 0
    t = step
    tm1 = (T - 1 - t - skipfirst)

    for k in range(skipfirst):
        step1 = s[t - step]
        x1 = (1.0 - leak) * x1 + leak * (inweights["U1"][:, step1] + cp.roll(x1, 1))
        x1 = x1 / cp.linalg.norm(x1)
        t += 1

    for b1 in range(tm1):
        step1 = s[t - step]

        x1 = (1.0 - leak) * x1 + leak * (inweights["U1"][:, step1] + cp.roll(x1, 1))
        x1 = x1 / cp.linalg.norm(x1)
        wx = cp.dot(variables["W1"], x1)
        wx = wx - cp.max(wx)
        p = cp.exp(wx)
        p1 = p / cp.sum(p)
        pred1 = cp.argmax(p1)

        target = s[t+1]
        target_prob1 = p1[target]
        softerr1 += 1 - target_prob1
        err1 = err1 + (pred1 != target)

        if testflag == 0:
            gradient["W1"] = cp.outer(v[:, target] - p1, x1)
            SGD.update_state(gradient)
            delta = SGD.get_delta()
            SGD.update_with_L1_regularization(variables, delta, L1)
        t += 1

    softerrors = dict()
    prederrors = dict()
    softerrors["lay1"] = softerr1 / (tm1)
    prederrors["lay1"] = err1 * 100.0 / (tm1)

    return prederrors, softerrors, variables

amath.setup(dycupy)
chunkfile = '/home/user01/dev/language-model/chunks256.p'

train1280 = '/home/user01/dev/language-model/train1280.p'
test128 = '/home/user01/dev/language-model/test128.p'

chunklist = pickle.load(open(chunkfile, "rb"))
layerscales = dict()
variables = dict()
inweights = dict()
L2 = dict()
L1 = dict()
steps = dict()
trainchunks = []
testchunks = []
cp.random.seed(481639)

n=2
stride = 1
# leaks = [0.382, 0.5, 0.618]
leak = 0.382
N = 1024
M = 1024

layerscales["L1"] = 3
# layerscales["L2"] = 3
# layerscales["L3"] = 2

## use 1-2-4-7-12-20-33-54-88, the fibonacci numbers look better as ##
## distances between points, rather than as the points themselves   ##

savedweights = 0
step = 0
batchsize = 1
trainsize = 64
testsize = 32
interval = 128
lrate = 0.002

SGD = ADAM(alpha=lrate)
variables["W1"] = cp.zeros((M, N * layerscales["L1"]), dtype=np.float32)
inweights["U1"] = cp.random.rand(N * layerscales["L1"], M, dtype=np.float32)

SGD = SGD.set_shape(variables)
for key in variables:
    L1[key] = 0
    L2[key] = 0

inweights, v = init(M, N, inweights)

layersize = str(inweights["U1"].shape[0])
print("L1: {}".format(layersize))
print("Learning rate:", lrate, "Batch size:", batchsize)
print("step: {}".format(step))

### load pre integer tokenized dataset of ~1 million characters in size
# trainfile = '/home/user01/dev/language-model/train1m.p'
# testfile = '/home/user01/dev/language-model/test1m.p'
# trainlist = pickle.load(open(trainfile, "rb"))
# testlist = pickle.load(open(testfile, "rb"))
#
# for chunk in trainlist:
#     intchunk = cp.array(chunk, dtype=np.int16)
#     trainchunks.append(intchunk)
#
# for chunk in testlist:
#     intchunk = cp.array(chunk, dtype=np.int16)
#     testchunks.append(intchunk)

for j in range(trainsize):
    chunk = chunklist[j]
    sgi = []
    for idx in range(0, len(chunk) - (n - 1), stride):
        try:
            sgi.append(gramindex[chunk[idx:idx + n]])
        except:
            print(chunk[idx:idx + n])
    intchunk = cp.asarray(sgi, dtype=np.int16)
    trainchunks.append(intchunk)

for k in range(trainsize, trainsize + testsize):
    chunk = chunklist[k]
    sgi = []
    for idx in range(0, len(chunk) - (n - 1), stride):
        try:
            sgi.append(gramindex[chunk[idx:idx + n]])
        except:
            print(chunk[idx:idx + n])
    intchunk = cp.asarray(sgi, dtype=np.int16)
    testchunks.append(intchunk)


trainsize = len(trainchunks)
testsize = len(testchunks)
print("train size:", trainsize, "test size:", testsize, "layersize:", layersize)
print(leak)

### get kernel PCA states
# cpstates = cp.empty((0, N * layerscales["L1"]), dtype=np.float32)
# npstates = np.empty((0, N * layerscales["L1"]), dtype=np.float32)
# totalerr1 = 0
# totalstates = 0
# testflag = 0
# count = 0
# totalstart = time.perf_counter()
#
# for chunk in trainchunks:
#     count += 1
#     startp = time.perf_counter()
#     variables, cpstates = train_kcpa(inweights, v, variables, leak, batchsize, step, chunk, cpstates)
#     npstates = np.concatenate((npstates, cp.asnumpy(cpstates)))
#     cpstates = cp.empty((0, N * layerscales["L1"]), dtype=np.float32)
#     totalstates += len(chunk) - 2
#     if count % interval == 0:
#         elapsedp = time.perf_counter() - startp
#         totalelapsed = time.perf_counter() - totalstart
#         tm, ts = divmod(totalelapsed, 60)
#         print("\n", count, elapsedp, "-- {0:.0f}m {1:.0f}s".format(tm, ts))
#         print("total states:", totalstates, "npstates:", npstates.shape)
#
# statefile = '/home/user01/dev/language-model/states' + layersize + "-" + str(step) + ".p"
# pickle.dump(npstates, open(statefile, "wb"))
# print("total states:", totalstates, "npstates:", npstates.shape)
# elapsedp = time.perf_counter() - startp
# totalelapsed = time.perf_counter() - totalstart
# tm, ts = divmod(totalelapsed, 60)
# print("\n", count, elapsedp, "-- {0:.0f}m {1:.0f}s".format(tm, ts))



# shuffle(trainchunks)

# print(inweights["U1"], variables["W1"] )

# outweights = '/home/user01/dev/language-model/outweights' + layersize + "-" + str(step) + ".p"
# inweights = '/home/user01/dev/language-model/inweights' + layersize + "-" + str(step) + ".p"


# saved_outweights = pickle.load(open(outweights, "rb"))
# saved_inweights = pickle.load(open(inweights, "rb"))
# print(saved_inweights["U1"].shape, type(saved_inweights["U1"]))
# print(saved_outweights["W1"].shape, type(saved_outweights["W1"]))

# inweights["U1"] = saved_inweights["U1"]
# variables["W1"] = saved_outweights["W1"]
# print(inweights["U1"], variables["W1"] )

######################################################################
lrs = str(lrate)
lrs = "-" + lrs[2:]
lrs = "-" + "500"
if savedweights == 1:
    # winfile = '/home/user01/dev/language-model/inweights' + layersize + "-" + str(step) + lrs + ".p"
    # woutfile = '/home/user01/dev/language-model/outweights' + layersize + "-" + str(step) + lrs + ".p"
    winfile = '/home/user01/dev/language-model/inweights8192-0-382' + ".p"
    woutfile = '/home/user01/dev/language-model/inweights8192-0-382' + ".p"
    print("U: {}\nW: {}".format(winfile, woutfile))
    saved_inweights = pickle.load(open(winfile, "rb"))
    saved_outweights = pickle.load(open(woutfile, "rb"))
    inweights["U1"] = saved_inweights["U1"]
    variables["W1"] = saved_outweights["W1"]
    print(saved_inweights["U1"].shape, saved_outweights["W1"].shape)
# shuffle(trainchunks)
# shuffle(testchunks)
# totalstart = time.perf_counter()
testflag = 0
count = 0
for i in range(64):
    epocherr1 = 0
    epochpred1 = 0
    totalerr1 = 0
    prederr1 = 0
    # istart = time.perf_counter()
    startp = time.perf_counter()

    for chunk in trainchunks:
        count += 1
        prederrs, softerrs, variables = train(inweights, v, variables, leak, batchsize, step, testflag, chunk, count)
        # prederr1 += prederrs["lay1"]
        # totalerr1 += softerrs["lay1"]
        epochpred1 += prederrs["lay1"]
        epocherr1 += softerrs["lay1"]
        # if count % interval == 0:
        #     elapsedp = time.perf_counter() - startp
        #     totalelapsed = time.perf_counter() - totalstart
        #     tm, ts = divmod(totalelapsed, 60)
        #     totalerr1 = totalerr1 * 100 / interval
        #     prederr1 = prederr1 / interval
        #     print("\n", i, count, "-- {0:.0f}m {1:.0f}s".format(tm, ts))
        #     print("Error: ", prederr1)
        #     print("Loss: ", totalerr1)
        #
        #     startp = time.perf_counter()
        #     totalerr1 = 0
        #     prederr1 = 0
    elapsedp = time.perf_counter() - startp
    tm, ts = divmod(elapsedp, 60)
    print("\n", i, count, "-- {0:.0f}m {1:.0f}s".format(tm, ts))
    epocherr1 = epocherr1 * 100 / trainsize
    epochpred1 = epochpred1 / trainsize
    print("Error: ", epochpred1)
    print("Loss: ", epocherr1)
    shuffle(trainchunks)

    if i > 0 and i % 128 == 0:
        totalerr1 = 0
        print("\n-----------\n Testing...\n-----------")
        testflag = 1

        for chunk in testchunks:
            prederrs, softerrs, variables = train(inweights, v, variables, leak, batchsize, step, testflag, chunk, count)
            totalerr1 += softerrs["lay1"]

        totalerr1 = totalerr1 * 100 / testsize
        print("Test Error:", prederrs["lay1"])
        print("Test Loss:", totalerr1)
        shuffle(testchunks)
        testflag = 0

    lrs = "-" + "5012301"
    winfile = '/home/user01/dev/language-model/inweights' + layersize + "-" + str(step) + lrs + ".p"
    woutfile = '/home/user01/dev/language-model/outweights' + layersize + "-" + str(step) + lrs + ".p"
    pickle.dump(inweights, open(winfile, "wb"))
    pickle.dump(variables, open(woutfile, "wb"))
