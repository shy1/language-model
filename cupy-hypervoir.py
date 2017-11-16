## Reservoir computing on the hypersphere
import numpy as np
import cupy as cp
import chargrams as cg
import re
import pickle
import time

#from sklearn.metrics import log_loss
from gensim.models.keyedvectors import KeyedVectors

import pydybm.arraymath as amath
import pydybm.arraymath.dycupy as dycupy
from pydybm.base.sgd32 import ADAM



def gramsintext(text, n=2):
    grams = cg.chargrams(text, n)
    glist = []
    for ngram, cnt in grams.items():
        glist.append(ngram)
    gramindex = {gram:idx for idx, gram in enumerate(glist)}
    return glist, gramindex

# create input weight matrix u and output value one-hot identity matrix(?) v
def init(M, N):
#    nu = np.empty((N, M), dtype=np.float32)
    ui = cp.random.rand(N * layerscales["L1"], M, dtype=np.float32)
    # u1 = cp.random.rand(N,M, dtype=np.float32)
    u2 = cp.random.rand(N * layerscales["L2"], M, dtype=np.float32)
    # u3 = cp.random.rand(N,M, dtype=np.float32)
    u4 = cp.random.rand(N * layerscales["L3"], M, dtype=np.float32)
    v = cp.identity(M, dtype=np.float32)
    # normalizing columns in NxM sized input matrix U as in formulas 6, 7
#    wv = KeyedVectors.load_word2vec_format('/home/user01/dev/wang2vec/embeddings-i3e4-ssg-neg15-s512w9.txt', binary=False)
    wv = KeyedVectors.load_word2vec_format('/home/user01/dev/wang2vec/embeddings-i3e4-ssg-neg15-s1024w6.txt', binary=False)
    temp = wv.index2word
    glist = np.array(temp[1:len(temp)])
#    print(len(arr))
#    print(wv['e_'])

#    for i in range(0, M):
#        temp = glist[i]
#        nu[:, i] = wv.word_vec(temp, use_norm=False)

    #print(nu.shape, nu)
#    u = cp.asarray(nu)

    for m in range(M):
        ui[:, m] = ui[:, m] - ui[:, m].mean()
        ui[:, m] = ui[:, m] / cp.linalg.norm(ui[:, m])

    # for m in range(M):
    #     u1[:, m] = u1[:, m] - u1[:, m].mean()
    #     u1[:, m] = u1[:, m] / cp.linalg.norm(u1[:, m])

    for m in range(M):
        u2[:, m] = u2[:, m] - u2[:, m].mean()
        u2[:, m] = u2[:, m] / cp.linalg.norm(u2[:, m])

    # for m in range(M):
    #     u3[:, m] = u3[:, m] - u3[:, m].mean()
    #     u3[:, m] = u3[:, m] / cp.linalg.norm(u3[:, m])

    for m in range(M):

        u4[:, m] = u4[:, m] - u4[:, m].mean()
        u4[:, m] = u4[:, m] / cp.linalg.norm(u4[:, m])
    #print(u.shape, u)
    glist = [re.sub(r'_', ' ', j) for j in glist]
    #print(glist)
    return ui, u2, u4, v, glist

def grecall(T, N, w, u, a, ss):
    x, i = cp.zeros(N, dtype=np.float32), ss
    ssa = []
    ssa.append(ss)
    for t in range(T - 1):
        x = (1.0 - a) * x + a * (u[:, i] + cp.roll(x, 1))
        x = x / cp.linalg.norm(x)
        y = cp.exp(cp.dot(w, x))
        i = cp.argmax(y / cp.sum(y))
        ssa.append(i)
    return ssa

def generate(T, N, u, variables, a, s0, temp=0.5):
    x, i = cp.zeros(N, dtype=np.float32), s0
    ssa = []
    ssa.append(s0)

    for t in range(T - 1):
        x = (1.0 - a) * x + a * (u[:, i] + cp.roll(x, 1))
        x = x / cp.linalg.norm(x)
        # probability distribution computed same as in online training
        # except that output of dot(w, x) is divided by the temperature
        output = cp.dot(variables["W1"], x) / temp
        output = output - np.max(output)
        probs = cp.exp(output)
        probs = probs / cp.sum(probs)
        i = cp.argmax(cp.random.multinomial(1, probs))
        ssa.append(i)

    sstext = ""
    for ssi in range(0, len(ssa), 2):
        #print(ssi, type(ssi))
        sstext += glist[int(ssa[ssi])]
    print(sstext)

def error(s, ss):
    err = 0.
    for t in range(len(s)):
        err = err + (s[t] != ss[t])
    #print(err)
    totalerr = err*100.0 / len(s)
    return totalerr

def online_grams(u, v, w, a, s):
    T, (N, M) = len(s), u.shape
    err = 100
    tt = 0

    #total = time.clock()
    totalp = time.perf_counter()
    while err > 0 and tt < T:
        x = cp.zeros(N, dtype=np.float32)
        softerr = 0
        for t in range(T - 1):
            #start = time.clock()
            #startp = time.perf_counter()
            x = (1.0 - a) * x + a * (u[:, s[t]] + cp.roll(x, 1))
            x = x / cp.linalg.norm(x)
            p = cp.exp(cp.dot(w, x))
#            psum = cp.sum(p)
            p = p / cp.sum(p)
#            print(p[0:19])
#            print(v[0:19, s[t+1]])
            smax_idx = cp.argmax(p)
            smax_prob = p[smax_idx]
            softerr += 1 - smax_prob
            w = w + cp.outer(v[:, s[t+1]] - p, x)
        avgerr = softerr / (T - 1)
        ssa = grecall(T, N, w, u, a, s[0])
        ssg = generate(100, N, u, a, s[0], s[1])
        err = error(s, ssa)

        tt = tt + 1
        if tt % 3 == 0:
#            a = a * 0.98
            #elapsed = time.clock() - start
            #elapsedp = time.perf_counter() - startp
            print(tt, "err:", err, "%", "softavg:", avgerr, "alpha:", a)
            sstext = ""
            for ssi in range(0, len(ssg), 2):
                #print(ssi, type(ssi))
                sstext += glist[int(ssg[ssi])]
            print(sstext)
    sstext = ""
    #endtotal = time.clock() - total
    endtotalp = time.perf_counter() - totalp
    for ssi in range(0, len(ssa), 2):
        #print(ssi, type(ssi))
        sstext += glist[int(ssa[ssi])]
    print(tt, "err=", err, "%\n", sstext, "\n", endtotalp)
    sstext = ""
    for ssi in range(0, len(ssg), 2):
        #print(ssi, type(ssi))
        sstext += glist[int(ssg[ssi])]
    print(sstext)
    return ssa, w

def offline_grams(u, v, c, a, s):
    sstext = ""
    T, (N, M), eta = len(s), u.shape, 1e-7
    X, S, x = cp.zeros((N, T-1), dtype=np.float32), cp.zeros((M, T-1), dtype=np.float32), cp.zeros(N, dtype=np.float32)

    # for j in range(0,3):
    #     print(j)
    for t in range(T - 1):
        x = (1.0 - a) * x + a * (u[:, s[t]] + cp.roll(x, 1))
        x = x / cp.linalg.norm(x)
        X[:, t], S[:, t] = x, v[:, s[t+1]]

    XX = cp.dot(X, X.T)
    for n in range(N):
        XX[n, n] = XX[n, n] + eta
    w = cp.dot(cp.dot(S, X.T), cp.linalg.inv(XX))
    ssa = grecall(T, N, w, u, c, alpha, s[0])

    sstext = ""
    for ssi in range(0, len(ssa), 2):
        sstext += glist[int(ssa[ssi])]
    print("err=", error(s, ssa), "%\n", sstext, "\n")
    return ssa, w

def train(ui, u2, u4, v, variables, leaks, s):
    gradient = dict()

    # bs = batch size
    bs = 32
    T = len(s)
    N = 1024
    M = 1024
    #total = time.clock()
    #totalp = time.perf_counter()
    x1 = cp.zeros(N * layerscales["L1"], dtype=np.float32)
    # x2 = cp.zeros(N, dtype=np.float32)
    x3 = cp.zeros(N * layerscales["L2"], dtype=np.float32)
    # x4 = cp.zeros(N, dtype=np.float32)
    x5 = cp.zeros(N * layerscales["L3"], dtype=np.float32)
    # print("x1: {} x3: {} x5: {}".format(x1.shape, x3.shape, x5.shape))

    softerr1 = 0
    err1 = 0
    softerr3 = 0
    err3 = 0
    softerr5 = 0
    err5 = 0
    softerrm = 0
    errm = 0
#    print(x1.shape, variables["W1"].shape)
    tm1 = (T - 1)

    # floored quotient (integer without remainder)
    fullbatches = tm1 // bs
    lastlen = tm1 - (fullbatches * bs)
    t = 0
    batchgrads1 = cp.empty(bs, dtype=np.float32)
    batchgrads3 = cp.empty(bs, dtype=np.float32)
    batchgrads5 = cp.empty(bs, dtype=np.float32)
    lastgrads1 = cp.empty(lastlen, dtype=np.float32)
    lastgrads3 = cp.empty(lastlen, dtype=np.float32)
    lastgrads5 = cp.empty(lastlen, dtype=np.float32)
    # print("batch: {} last:{}".format(batchgrads1.shape, lastgrads1.shape))
    for b in range(fullbatches):

        for i in range(bs):
            current = s[t]

            x1 = (1.0 - leaks[0]) * x1 + leaks[0] * (ui[:, current] + cp.roll(x1, 1))
            x1 = x1 / cp.linalg.norm(x1)
            wx = cp.dot(variables["W1"], x1)
            wx = wx - cp.max(wx)
            p = cp.exp(wx)
            p1 = p / cp.sum(p)

            pu2 = cp.dot(p1, u2.T)
            x3 = (1.0 - leaks[1]) * x3 + leaks[1] * (pu2 + cp.roll(x3, 1))
            x3 = x3 / cp.linalg.norm(x3)
            wx = cp.dot(variables["W3"], x3)
            wx = wx - cp.max(wx)
            p = cp.exp(wx)
            p3 = p / cp.sum(p)

            pu4 = cp.dot(p3, u4.T)
            x5 = (1.0 - leaks[2]) * x5 + leaks[2] * (pu4 + cp.roll(x5, 1))
            x5 = x5 / cp.linalg.norm(x5)
            wx = cp.dot(variables["W5"], x5)
            wx = wx - cp.max(wx)
            p = cp.exp(wx)
            p5 = p / cp.sum(p)
            pred5 = cp.argmax(p5)

            # pstack = cp.stack((p1, p3, p5), axis=1)
            # print(variables["Ws"].shape, pstack.shape)

            # wx = cp.dot(variables["Ws"], wxstack)
            # # wx = cp.dot(pstack, variables["Ws"])
            # # print(wx.shape)
            # wx = wx - cp.max(wx)
            # p = cp.exp(wx)
            # pm = p / cp.sum(p)
            # meanpred = cp.argmax(pm)

            target = s[t+1]

            target_prob1 = p1[target]
            target_prob3 = p3[target]
            target_prob5 = p5[target]
            err5 = err5 + (pred5 != target)

            softerr1 += 1 - target_prob1
            softerr3 += 1 - target_prob3
            softerr5 += 1 - target_prob5

            batchgrads1[i] = cp.outer(v[:, target] - p1, x1)
            batchgrads3[i] = cp.outer(v[:, target] - p3, x3)
            batchgrads5[i] = cp.outer(v[:, target] - p5, x5)

            t += 1

        gradient["W1"] = batchgrads1.mean()
        gradient["W3"] = batchgrads3.mean()
        gradient["W5"] = batchgrads5.mean()
        SGD.update_state(gradient)
        delta = SGD.get_delta()
        SGD.update_with_L1_regularization(variables, delta, L1)
        # SGD.apply_L2_regularization(gradient, variables, L2)
    for j in range(lastlen):
        current = s[t]
        x1 = (1.0 - leaks[0]) * x1 + leaks[0] * (ui[:, current] + cp.roll(x1, 1))
        x1 = x1 / cp.linalg.norm(x1)
        wx = cp.dot(variables["W1"], x1)

        wx = wx - cp.max(wx)
        p = cp.exp(wx)
        p1 = p / cp.sum(p)

        pu2 = cp.dot(p1, u2.T)
        x3 = (1.0 - leaks[1]) * x3 + leaks[1] * (pu2 + cp.roll(x3, 1))
        x3 = x3 / cp.linalg.norm(x3)
        wx = cp.dot(variables["W3"], x3)
        wx = wx - cp.max(wx)
        p = cp.exp(wx)
        p3 = p / cp.sum(p)

        pu4 = cp.dot(p3, u4.T)
        x5 = (1.0 - leaks[2]) * x5 + leaks[2] * (pu4 + cp.roll(x5, 1))
        x5 = x5 / cp.linalg.norm(x5)
        wx = cp.dot(variables["W5"], x5)
        wx = wx - cp.max(wx)
        p = cp.exp(wx)
        p5 = p / cp.sum(p)
        pred5 = cp.argmax(p5)

        target = s[t+1]
        target_prob1 = p1[target]
        target_prob3 = p3[target]
        target_prob5 = p5[target]
        err5 = err5 + (pred5 != target)

        softerr1 += 1 - target_prob1
        softerr3 += 1 - target_prob3
        softerr5 += 1 - target_prob5

        lastgrads1[j] = cp.outer(v[:, target] - p1, x1)
        lastgrads3[j] = cp.outer(v[:, target] - p3, x3)
        lastgrads5[j] = cp.outer(v[:, target] - p5, x5)

        t += 1
    if lastlen > 0:
        gradient["W1"] = lastgrads1.mean()
        gradient["W3"] = lastgrads3.mean()
        gradient["W5"] = lastgrads5.mean()
        SGD.update_state(gradient)
        delta = SGD.get_delta()
        SGD.update_with_L1_regularization(variables, delta, L1)

    softerrors = dict()
    prederrors = dict()
    softerrors["lay1"] = softerr1 / (tm1)
    softerrors["lay3"] = softerr3 / (tm1)
    softerrors["lay5"] = softerr5 / (tm1)
    # softerrors["laym"] = softerrm / (tm1)
    # prederrors["lay1"] = err1 * 100.0 / (tm1)
    # prederrors["lay3"] = err3 * 100.0 / (tm1)
    prederrors["lay5"] = err5 * 100.0 / (tm1)
    # prederrors["laym"] = errm * 100.0 / (tm1)
    return prederrors, softerrors, variables


n=2
stride = 1


#s = re.sub('\n', ' ', s)
#if unclean:
#    s = cg.gut_clean(s)

cp.random.seed(481639)
## alpha of .4452 achieved 0.0 err with offline grams using 1-d512-w9 and 946 char document

#alpha = .369
#alpha = .4452
#alpha = .467
#alpha = .487
# alpha = .52037
#alpha = .537
#alpha = .58
#alpha = 0.6301

# leaks = [0.8155, 0.6309, 0.3155]
# leaks = [0.3155, 0.6309, 0.8155]
leaks = [0.6905, 0.5655, 0.4405]
# leaks = [0.6905, 0.581, 0.4405]
# leaks = [0.4405, 0.5655, 0.6905]
# leaks = [0.73814, .52037, 0.3679]
# leaks = [0.73814, .58078, 0.43178]
N = 1024
#N = 1024
M = 1024
layerscales = dict()
layerscales["L1"] = 3
layerscales["L2"] = 2
layerscales["L3"] = 3

amath.setup(dycupy)
lrate = 0.0025
SGD = ADAM(alpha=lrate)

variables = dict()
variables["W1"] = cp.zeros((M, N * layerscales["L1"]), dtype=np.float32)
variables["W3"] = cp.zeros((M, N * layerscales["L2"]), dtype=np.float32)
variables["W5"] = cp.zeros((M, N * layerscales["L3"]), dtype=np.float32)
# variables["Ws"] = cp.zeros((1024, N*3), dtype=np.float32)
# variables["Ws"] = cp.array([0.002, 0.003, 0.005], dtype=np.float32)
SGD = SGD.set_shape(variables)
print("Learning rate:", SGD.alpha)
L2 = dict()
L1 = dict()
for key in variables:
    L1[key] = 0
    L2[key] = 0

ui, u2, u4, v, glist = init(M, N)
gramindex = {gram:idx for idx, gram in enumerate(glist)}
# print(len(gramindex), type(gramindex))
print("ui: {} u2: {} u4: {}".format(ui.shape, u2.shape, u4.shape))
print("W1: {} W3: {} W5: {}".format(variables["W1"].shape, variables["W3"].shape, variables["W5"].shape))

chunkfile = '/home/user01/dev/language-model/chunks100mb.p'
outweights = '/home/user01/dev/language-model/outweights100mb.p'
inweights = '/home/user01/dev/language-model/inweights100mb.p'
#pickle.dump(allchunks, open(outfile, "wb"))
chunklist = pickle.load(open(chunkfile, "rb"))
print(len(chunklist))
intchunklist = []

for j in range(33):
    chunk = chunklist[j]
    # pad with space if chunk has an odd number of characters so that
    # chunk length is compatibile with bigrams using a stride of 2
    if len(chunk) % 2 != 0:
        chunk += " "
    sgi = []
    for idx in range(0, len(chunk) - (n - 1), stride):
        try:
            sgi.append(gramindex[chunk[idx:idx + n]])
        except:
            print(chunk[idx:idx + n])
    intchunk = cp.asarray(sgi, dtype=np.int16)
    intchunklist.append(intchunk)

#for i in range(len(chunklist))
print(leaks[0], leaks[1], leaks[2])
totalstart = time.perf_counter()
for i in range(99):
    startp = time.perf_counter()
    totalerr1 = 0
    totalerr3 = 0
    totalerr5 = 0
    totalerrm = 0
    for chunk in intchunklist:
        prederrs, softerrs, variables = train(ui, u2, u4, v, variables, leaks, chunk)
        totalerr1 += softerrs["lay1"]
        totalerr3 += softerrs["lay3"]
        totalerr5 += softerrs["lay5"]
        # totalerrm += softerrs["laym"]
#        if i % 100 == 0:
    elapsedp = time.perf_counter() - startp
    totalelapsed = time.perf_counter() - totalstart
    tm, ts = divmod(totalelapsed, 60)
    totalerr1 = totalerr1 / 33
    totalerr3 = totalerr3 / 33
    totalerr5 = totalerr5 / 33
    # totalerrm = totalerrm / 33
    print("\n", i, elapsedp, "-- {0:.0f}m {1:.0f}s".format(tm, ts))
    # print("Errors:", prederrs["lay1"], prederrs["lay3"], prederrs["lay5"])
    print("Errors:", prederrs["lay5"])
    print("Losses:", totalerr1, totalerr3, totalerr5)
    # print("Losses:", totalerr5)
    # generate(128, N, u, variables, alpha, 4)

pickle.dump(variables, open(outweights, "wb"))
pickle.dump(ui, open(inweights, "wb"))


#    if tt % 10 == 0:
##        a = a * 0.98
#
#        print(tt, "err:", err, "%", "softavg:", avgerr, "alpha:", a)
#        sstext = ""
#        for ssi in range(0, len(ssg), 2):
#            #print(ssi, type(ssi))
#            sstext += glist[int(ssg[ssi])]
#        print(sstext)


#N = int(M * 2.71828)
#div = ' mt:nt '
#nt = N / T
#nmt = [str(mt), str(nt)]
#print(div.join(nmt))
#print(T, N, M, alpha)
#print(T, N, M, alpha)

#ss, w = offline_grams(u, v, glist, alpha, sgi)

# totalo = time.perf_counter()
# endtotalo = time.perf_counter() - totalo
# print(endtotalo)
#cu_glist = cp.asarray(glist, dtype=np.float32)
#cu_sgi = cp.asarray(sgi, dtype=np.int16)

# tnt = int(T * 2)
# if tnt > M:
#     N = tnt
# else:
#     N = M

# T = length of sequence
# M = number of distinct input states
#T, M = len(s), len(allchars)
# mt = ratio of possible input states to timesteps to memorize
# nt = ratio of reservoir units to timesteps to memorize
# N = number of hidden states/reservoir size
# alpha = integration rate indicating "leakiness", 1 = no leaking/orig model
# alpha is determined by ratio nt (hidden states) minus ratio mt (input states)
# u = input weight matrix
# v = hidden state orthogonal identity matrix
#with open('test02.txt', 'r') as ofile:
#    s = ofile.read()

#with open('counts1024b.txt', 'r') as countfile:
#    counts = countfile.readlines()
#for line in counts:
#    glist.append(line[:2])
#gramindex = {gram:idx for idx, gram in enumerate(glist)}
#print(len(gramindex), type(gramindex))

#unclean = False

#number of characters in each n-gram sequence that forms a single unit/timestep
