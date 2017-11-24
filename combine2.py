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
    x1 = cp.zeros(int(N * layerscales["L1"]), dtype=np.float32)
    x2 = cp.zeros(int(N * layerscales["L1"]), dtype=np.float32)
    x3 = cp.zeros(int(N * layerscales["L1"]), dtype=np.float32)
    gradient = dict()
    softerr1 = 0
    softerr2 = 0
    softerr3 = 0
    softerrm = 0
    err1 = 0
    err2 = 0
    err3 = 0
    errm = 0
    skipfirst = 1
    t = steps["S2"]
    tm1 = (T - 1 - t - skipfirst)

    for k in range(skipfirst):
        step1 = s[t - steps["S1"]]
        step2 = s[t - steps["S2"]]
        x1 = (1.0 - leaks[0]) * x1 + leaks[0] * (inweights["U1"][:, step1] + cp.roll(x1, 1))
        x1 = x1 / cp.linalg.norm(x1)
        x2 = (1.0 - leaks[1]) * x2 + leaks[1] * (inweights["U2"][:, step2] + cp.roll(x2, 1))
        x2 = x2 / cp.linalg.norm(x2)
        t += 1

    for b1 in range(tm1):
        step1 = s[t - steps["S1"]]
        step2 = s[t - steps["S2"]]

        x1 = (1.0 - leaks[0]) * x1 + leaks[0] * (inweights["U1"][:, step1] + cp.roll(x1, 1))
        x1 = x1 / cp.linalg.norm(x1)
        wx = cp.dot(variables["W1"], x1)
        wx = wx - cp.max(wx)
        p = cp.exp(wx)
        p1 = p / cp.sum(p)
        # pred1 = cp.argmax(p1)

        x2 = (1.0 - leaks[1]) * x2 + leaks[1] * (inweights["U2"][:, step2] + cp.roll(x2, 1))
        x2 = x2 / cp.linalg.norm(x2)
        wx = cp.dot(variables["W2"], x2)
        wx = wx - cp.max(wx)
        p = cp.exp(wx)
        p2 = p / cp.sum(p)
        # pred2 = cp.argmax(p2)

        pstack = cp.hstack((p1, p2))
        wx = cp.dot(variables["Wm"], pstack)
        wx = wx - cp.max(wx)
        p = cp.exp(wx)
        pm = p / cp.sum(p)
        meanpred = cp.argmax(pm)

        target = s[t+1]
        target_prob1 = p1[target]
        softerr1 += 1 - target_prob1
        # err1 = err1 + (pred1 != target)
        target_prob2 = p2[target]
        softerr2 += 1 - target_prob2
        # err2 = err2 + (pred2 != target)
        target_probm = pm[target]
        errm = errm + (meanpred != target)
        softerrm += 1 - target_probm

        if testflag == 0:
            gradient["W1"] = cp.outer(v[:, target] - p1, x1)
            # gradient["W2"] = cp.outer(v[:, target] - p2, x2)
            gradient["Wm"] = cp.outer(v[:, target] - pm, pstack)
            SGD.update_state(gradient)
            delta = SGD.get_delta()
            SGD.update_with_L1_regularization(variables, delta, L1)
        t += 1

    softerrors = dict()
    prederrors = dict()
    softerrors["lay1"] = softerr1 / (tm1)
    softerrors["lay2"] = softerr2 / (tm1)
    softerrors["laym"] = softerrm / (tm1)
    # prederrors["lay1"] = err1 * 100.0 / (tm1)
    # prederrors["lay2"] = err2 * 100.0 / (tm1)
    prederrors["laym"] = errm * 100.0 / (tm1)
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
leaks = [0.382, 0.5]
# leak = 0.382
N = 1024
M = 1024

layerscales["L1"] = 8
# layerscales["L2"] = 3
# layerscales["L3"] = 2

## use 1-2-4-7-12-20-33-54-88, the fibonacci numbers look better as ##
## distances between points, rather than as the points themselves   ##

savedweights = 1
steps["S1"] = 0
steps["S2"] = 2
batchsize = 1
trainsize = 256
testsize = 32
interval = 128
lrate = 0.0001
learningrates = [0.09, 0.03, 0.009, 0.003, 0.0009, 0.0003, 0.0001]
SGD = ADAM(alpha=lrate)
# SGD = ADAM(alpha=learningrates[0])

variables["W1"] = cp.zeros((M, int(N * layerscales["L1"])), dtype=np.float32)
variables["W2"] = cp.zeros((M, int(N * layerscales["L1"])), dtype=np.float32)
variables["Wm"] = cp.zeros((M, M*2), dtype=np.float32)
inweights["U1"] = cp.random.rand(int(N * layerscales["L1"]), M, dtype=np.float32)
inweights["U2"] = cp.random.rand(int(N * layerscales["L1"]), M, dtype=np.float32)
allweights = dict()

SGD = SGD.set_shape(variables)
for key in variables:
    L1[key] = 0
    L2[key] = 0

inweights, v = init(M, N, inweights)

layersize1 = str(inweights["U1"].shape[0])
layersize2 = str(inweights["U2"].shape[0])
print("L1: {} L2: {}".format(layersize1, layersize2))
print("steps: {} {}".format(steps["S1"], steps["S2"]))

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
print("train size:", trainsize, "test size:", testsize)
print("Learning rate:", lrate)
print(leaks[0], leaks[1])

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
lay1infile = '/home/user01/dev/language-model/inweights8192-0-382' + ".p"
lay2infile = '/home/user01/dev/language-model/inweights8192-2-500' + ".p"
lay1outfile = '/home/user01/dev/language-model/outweights8192-0-382' + ".p"
lay2outfile = '/home/user01/dev/language-model/outweights8192-2-500' + ".p"

# print("U1: {}\nW1: {}\nU2: {}\nW2: {}".format(lay1infile, lay2infile, lay1outfile, lay2outfile))
# lay1_inweights = pickle.load(open(lay1infile, "rb"))
# lay1_outweights = pickle.load(open(lay1outfile, "rb"))
# lay2_inweights = pickle.load(open(lay2infile, "rb"))
# lay2_outweights = pickle.load(open(lay2outfile, "rb"))
# inweights["U1"] = lay1_inweights["U1"]
# inweights["U2"] = lay2_inweights["U1"]
# variables["W1"] = lay1_outweights["W1"]
# variables["W2"] = lay2_outweights["W1"]
# variables["Wm"] = saved_outweights["Wm"]
# print(lay1_inweights["U1"].shape, lay2_outweights["W1"].shape)

# lrs = str(lrate)
# lrs = "-" + lrs[2:]
stepstr = str(steps["S1"]) + str(steps["S2"])
weightfile = '/home/user01/dev/language-model/weights2.' + layersize1 + '.0-2.382-500.p'
if savedweights == 1:
    weightfile = '/home/user01/dev/language-model/weights2.' + layersize1 + '.0-2.382-500.p'

    print("W: {}".format(weightfile))
    saved_weights = pickle.load(open(weightfile, "rb"))
    inweights["U1"] = saved_weights["U1"]
    inweights["U2"] = saved_weights["U2"]
    variables["W1"] = saved_weights["W1"]
    variables["W2"] = saved_weights["W2"]
    variables["Wm"] = saved_weights["Wm"]
    print(saved_weights["U1"].shape, saved_weights["W2"].shape)
# shuffle(trainchunks)
# shuffle(testchunks)
totalstart = time.perf_counter()
testflag = 0
count = 0
# for lr in range(7):
    # SGD.set_alpha(learningrates[lr])
    # print("Learning rate:", learningrates[lr])
for i in range(16):
    epocherrm = 0
    epochpredm = 0
    totalerr1 = 0
    totalerr2 = 0
    totalerrm = 0
    prederrm = 0
    # istart = time.perf_counter()
    startp = time.perf_counter()

    for chunk in trainchunks:
        count += 1
        prederrs, softerrs, variables = train(inweights, v, variables, leaks, batchsize, steps, testflag, chunk, count)
        # prederr1 += prederrs["lay1"]
        # totalerr1 += softerrs["lay1"]
        epochpredm += prederrs["laym"]
        totalerr1 += softerrs["lay1"]
        totalerr2 += softerrs["lay2"]
        epocherrm += softerrs["laym"]
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
    totalelapsed = time.perf_counter() - totalstart
    em, es = divmod(elapsedp, 60)
    tm, ts = divmod(totalelapsed, 60)
    print("\n", i, count, "{0:.0f}m {1:.0f}s -- {2:.0f}m {3:.0f}s".format(em, es, tm, ts))
    totalerr1 = totalerr1 * 100 / trainsize
    totalerr2 = totalerr2 * 100 / trainsize
    epocherrm = epocherrm * 100 / trainsize
    epochpredm = epochpredm / trainsize
    print("Error: ", epochpredm)
    print("Loss: ", totalerr1, totalerr2, "m:", epocherrm)
    shuffle(trainchunks)

    if (i + 1) % 16 == 0:
        totalerrm = 0
        print("\n-----------\n Testing...\n-----------")
        testflag = 1

        for chunk in testchunks:
            prederrs, softerrs, variables = train(inweights, v, variables, leaks, batchsize, steps, testflag, chunk, count)
            totalerrm += softerrs["laym"]

        totalerrm = totalerrm * 100 / testsize
        print("Test Error:", prederrs["laym"])
        print("Test Loss:", totalerrm)
        shuffle(testchunks)
        testflag = 0

    allweights["U1"] = inweights["U1"]
    allweights["U2"] = inweights["U2"]
    allweights["W1"] = variables["W1"]
    allweights["W2"] = variables["W2"]
    allweights["Wm"] = variables["Wm"]
    weightfile = '/home/user01/dev/language-model/weights2.' + layersize1 + '.0-2.382-500.p'
    # winfile = '/home/user01/dev/language-model/inweights3-' + layersize1 + "-" + stepstr + lrs + ".p"
    # woutfile = '/home/user01/dev/language-model/outweights3-' + layersize1 + "-" + stepstr + lrs + ".p"
    pickle.dump(allweights, open(weightfile, "wb"))
    # pickle.dump(variables, open(woutfile, "wb"))
