# Reservoir computing on the hypersphere
import numpy as np
import cupy as cp
import chargrams as cg
import csv
import time

def gramsintext(text, n=2):
    grams = cg.chargrams(text, n)
    glist = []
    for ngram, cnt in grams.items():
        glist.append(ngram)
    gramindex = {gram:idx for idx, gram in enumerate(glist)}
    return glist, gramindex

# create input weight matrix u and hidden state/reservoir matrix v
def init(M, N):
    u, v = cp.random.rand(N, M, dtype=np.float32), cp.identity(M, dtype=np.float32)
    # normalizing columns in NxM sized input matrix U as in formulas 6, 7

    for m in range(M):
        u[:, m] = u[:, m] - u[:, m].mean()
        u[:, m] = u[:, m] / cp.linalg.norm(u[:, m])
    return u, v

def grecall(T, N, w, u, c, a, ss):
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

def error(s, ss):
    err = 0.
    for t in range(len(s)):
        err = err + (s[t] != ss[t])
    #print(err)
    totalerr = err*100.0 / len(s)
    return totalerr

def online_grams(u, v, c, a, s):
    T, (N, M) = len(s), u.shape
    w = cp.zeros((M, N), dtype=np.float32)
    err = 100
    tt = 0
    #total = time.clock()
    totalp = time.perf_counter()
    while err > 0 and tt < T:
        x = cp.zeros(N, dtype=np.float32)

        for t in range(T - 1):
            #start = time.clock()
            #startp = time.perf_counter()
            x = (1.0 - a) * x + a * (u[:, s[t]] + cp.roll(x, 1))
            x = x / cp.linalg.norm(x)
            p = cp.exp(cp.dot(w, x))
            p = p / cp.sum(p)
            w = w + cp.outer(v[:, s[t+1]] - p, x)
        ssa = grecall(T, N, w, u, c, a, s[0])
        err = error(s, ssa)
        tt = tt + 1
        if tt % 3 == 0:
            a = a * 0.985
            #elapsed = time.clock() - start
            #elapsedp = time.perf_counter() - startp
            print(tt, "err=", err, "%", "alpha:", a)
    sstext = ""
    #endtotal = time.clock() - total
    endtotalp = time.perf_counter() - totalp
    for ssi in range(0, len(ssa), 2):
        #print(ssi, type(ssi))
        sstext += glist[int(ssa[ssi])]
    print(tt, "err=", err, "%\n", sstext, "\n", endtotalp)
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

    for ssi in ssa:
        sstext += glist[int(ssi)]
    print("err=", error(s, ssa), "%\n", sstext, "\n")
    return ssa, w


n=2
with open('test02.txt', 'r') as ofile:
    s = ofile.read()

glist = []
with open('counts1024b.txt', 'r') as countfile:
    counts = countfile.readlines()
for line in counts:
    glist.append(line[:2])
gramindex = {gram:idx for idx, gram in enumerate(glist)}
print(len(gramindex), type(gramindex))

unclean = False
sgi = []
stride = 1

if unclean:
    s = cg.gut_clean(s)

for idx in range(0, len(s) - (n - 1), stride):
    sgi.append(gramindex[s[idx:idx + n]])
print(len(sgi))

T, M = len(sgi), len(glist)
mt = M / T
N = int(M * 2.71828)
#N = 2560
#alpha = .52037
alpha = 0.53
#alpha = np.log(2) / np.log(3)
div = ' mt:nt '
nt = N / T
nmt = [str(mt), str(nt)]
print(div.join(nmt))
print(T, N, M, alpha)

cp.random.seed(35712)
u, v = init(M, N)
print(u.shape, v.shape)

# totalo = time.perf_counter()
# ss, w = offline_grams(u, v, glist, alpha, sgi)
# endtotalo = time.perf_counter() - totalo
# print(endtotalo)

ss, w = online_grams(u, v, glist, alpha, sgi)
print(T, N, M, alpha)

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
