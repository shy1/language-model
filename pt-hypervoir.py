# pytorch implementation of:
# "Reservoir Computing on the Hypersphere" - M. Andrecut arXiv:1706.07896
import torch
from torch.autograd import Variable
import torch.nn.functional as F
# Reservoir computing on the hypersphere
import numpy as np
import chargrams as cg
import csv

# create input weight matrix u and hidden state/reservoir matrix v
def initcuda(M,N):
    u = np.random.rand(N, M).astype(np.float32)
    v = np.identity(M)
    #u = torch.rand(N, M)
    #v = torch.eye(M, M)
    # normalizing columns in NxM sized input matrix U as in formulas 6,7
    for m in range(M):
        u[:,m] = u[:,m] - u[:,m].mean()
        u[:,m] = u[:,m]/np.linalg.norm(u[:,m])
    tv = torch.from_numpy(v)
    tu = torch.from_numpy(u)

    tv = tv.float().cuda()
    tu = tu.cuda()
    #print(tv)
    #print(tu)

    return tu, tv, u, v

def shiftcuda(matrix):
    #print(matrix)
    length = len(matrix)
    end = torch.cuda.FloatTensor([matrix[length - 1]])
    matrix = torch.cuda.FloatTensor(matrix.narrow(0, 0, length - 1))
    #print(end)
    #print(matrix)
    matrix = torch.cat((end, matrix), 0)
    return matrix

def grecall(T,N,w,tu,c,a,ss):
    x,i = np.zeros(N), ss
    #tx, i = torch.zeros(N), ss
    tx = torch.from_numpy(x).float().cuda()
    #y = torch.cuda.FloatTensor()
    #print(i)
    ssa = [ss]
    values = tp = torch.cuda.FloatTensor()
    indexes = torch.cuda.LongTensor()
    for t in range(T-1):
        #print(t)
        tx = (1 - a) * tx + a * (tu[:, i] + shiftcuda(tx))
        print(tx)
        tx = tx / tx.norm()
        y = torch.exp(torch.mm(tx.view(-1, M), w))
        #print(y)
        torch.max(y / torch.sum(y), 1, out=(values, indexes))
        i = (torch.max(indexes))
        #print(i)
        ssa.append(i)
    return ssa

def error(s, ss):
    #print(s, len(ss))
    err = 0.
    for t in range(len(s)):
        err = err + (s[t] != ss[t])
    return np.round(err*100.0/len(s),2)

def online_grams(tu, tv, c, a, s:
    sstext = ""
    T, (N, M) = len(s), tu.shape
    w,err,tt = np.zeros((M,N)),100.,0
    tw = torch.from_numpy(w).float().cuda()



    while err > 0 and tt < T:
        x = np.zeros(N).astype(np.float32)
        tx = torch.from_numpy(x)

        tx = tx.cuda()

        for t in range(T-1):
            #x = (1.0-a)*x + a*(u[:,s[t]] + np.roll(x,1))
            #print(x)
            tx = (1 - a) * tx + a * (tu[:, s[t]] + shiftcuda(tx))
            #print(tx)
            tx = tx / tx.norm()

            #print('w then tx')
            #print(w, tx)
            # print(N, M)
            # print(tx, w)


            p = torch.exp(torch.mm(tx.view(-1, M), tw))

            px = np.exp(np.dot(w,nx))

            #tp = torch.from_numpy(p).float().cuda()
            #print(p)
            p = p / torch.sum(p)

            p = p.squeeze()

            tw = tw + torch.ger(tv[:, s[t+1]] - p, tx)


            # szero = [s.select(0,0)]
            # print(szero)
        ssa = grecall(T, N, tw, tu, c, a, s[0])
        err, tt = error(s, ssa), tt+1

        if tt % 3 == 0:
            # for ssi in ssa:
            #     sstext += glist[ssi]
            # print(tt,"err=",err,"%\n",sstext,"\n")
            print(tt,"err=",err,"%")
    for ssi in ssa:
        sstext += glist[ssi]
    print(tt,"err=",err,"%\n",sstext,"\n")
    return ssa, w

def offline_grams(u,v,c,a,s):
    sstext = ""
    T,(N,M),eta = len(s),u.shape,1e-7
    X,S,x = np.zeros((N,T-1)),np.zeros((M,T-1)),np.zeros(N)
    for t in range(T-1):
        x = (1.0-a)*x + a*(u[:,s[t]] + np.roll(x,1))
        x = x/np.linalg.norm(x)
        X[:,t],S[:,t] = x,v[:,s[t+1]]
    XX = np.dot(X,X.T)
    for n in range(N):
        XX[n,n] = XX[n,n] + eta
    w = np.dot(np.dot(S,X.T),np.linalg.inv(XX))
    ssa = grecall(T,N,w,u,c,alpha,s[0])

    for ssi in ssa:
        sstext += glist[ssi]
    print("err=",error(s,ssa),"%\n",sstext,"\n")
    return ssa,w

s1 = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer--excellent for drawing the veil from men’s motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results. "
s2 = "Grit in a sensitive instrument, or a crack in one of his own high-power lenses, would not be more disturbing than a strong emotion in a nature such as his. And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory."
#s = s1 + s2
with open('test02.txt', 'r') as ofile:
    s = ofile.read()
# number of characters in n-gram
n=2

glist = []
with open('counts1024.txt', 'r') as countfile:
    counts = countfile.readlines()
for line in counts:
    glist.append(line[:2])
gramindex = {gram:idx for idx,gram in enumerate(glist)}
#print(len(gramindex))

temp = []
sgi = torch.IntTensor()
sg = cg.gut_clean(s)
#print(sgi)
for idx in range(0, len(sg) - (n - 1), 2):
    temp.append(gramindex[sg[idx:idx + n]])
sgi = torch.IntTensor(temp)
#print(len(sgi))
#print(sgi)

# T = length of sequence
# M = number of distinct input states
#T,M = len(s),len(allchars)
T,M = len(sgi),len(glist)

# N = number of hidden states/reservoir size (determined by ratio nt calculated above)
# alpha = integration rate indicating "leakiness", 1 = no leaking/orig model
# alpha is determined by ratio nt (hidden states) minus ratio mt (input states)

tnt = int(T * 0.62)
maxn = [tnt, M]
N = 1024
# mt = ratio of possible input states to timesteps to memorize
mt = M / T
# nt = ratio of reservoir units to timesteps to memorize
nt = N / T
alpha = 0.58
nmta = [str(mt), str(nt), str(alpha)]
div = ' - '
print(div.join(nmta))
print(M, N, alpha)


np.random.seed(11712)
# u = input weight matrix
# v = hidden state orthogonal identity matrix
tu,tv,u,v = initcuda(M,N)

#ss,w = offline_grams(u,v,glist,alpha,sgi)
ss,w = online_grams(tu,tv,glist,alpha,sgi)

print(T,N,M,alpha)
