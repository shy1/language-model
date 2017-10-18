# Reservoir computing on the hypersphere
import numpy as np
import chargrams as cg
import csv
import time

# create input weight matrix u and hidden state/reservoir matrix v
def init(M,N):
    u,v = np.random.rand(N,M),np.identity(M)
    # normalizing columns in NxM sized input matrix U as in formulas 6,7
    for m in range(M):
        u[:,m] = u[:,m] - u[:,m].mean()
        u[:,m] = u[:,m]/np.linalg.norm(u[:,m])
    return u,v
##
def recall(T,N,w,u,c,a,ss):
    x,i = np.zeros(N),characterindex[ss]
    for t in range(T-1):
        x = (1.0-a)*x + a*(u[:,i] + np.roll(x,1))
        x = x/np.linalg.norm(x)
        y = np.exp(np.dot(w,x))
        i = np.argmax(y/np.sum(y))
        ss = ss + str(c[i])
    return ss

def grecall(T,N,w,u,c,a,ss):
    x,i = np.zeros(N), ss
    ssa = [ss]
    for t in range(T-1):
        x = (1.0-a)*x + a*(u[:,i] + np.roll(x,1))
        x = x/np.linalg.norm(x)
        y = np.exp(np.dot(w,x))
        i = np.argmax(y/np.sum(y))
        ssa.append(i)
    return ssa

def error(s,ss):
    err = 0.
    for t in range(len(s)):
        err = err + (s[t]!=ss[t])
    return np.round(err*100.0/len(s),2)

def online_grams(u,v,c,a,s):
    sstext = ""
    T,(N,M) = len(s),u.shape
    w,err,tt = np.zeros((M,N)),100.,0
    totalp = time.perf_counter()
    while err>0 and tt<T:
        x = np.zeros(N)

        for t in range(T-1):
            #start = time.clock()
            #startp = time.perf_counter()
            x = (1.0-a)*x + a*(u[:,s[t]] + np.roll(x,1))
            x = x/np.linalg.norm(x)
            p = np.exp(np.dot(w,x))
            p = p/np.sum(p)
            w = w + np.outer(v[:,s[t+1]]-p,x)
        ssa = grecall(T,N,w,u,c,a,s[0])
        err,tt = error(s,ssa),tt+1

        if tt % 3 == 0:
            # sstext = ""
            # for ssi in ssa:
            #     sstext += glist[ssi]
            # print(tt,"err=",err,"%\n",sstext,"\n")
            #elapsed = time.clock() - start
            #elapsedp = time.perf_counter() - startp
            print(tt,"err=",err,"%")
    endtotalp = time.perf_counter() - totalp
    for ssi in ssa:
        sstext += glist[ssi]
    print(tt,"err=",err,"%\n",sstext,"\n", endtotalp)
    return ssa,w

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

def offline_learning(u,v,c,a,s):
    T,(N,M),eta = len(s),u.shape,1e-7
    X,S,x = np.zeros((N,T-1)),np.zeros((M,T-1)),np.zeros(N)
    for t in range(T-1):
        x = (1.0-a)*x + a*(u[:,characterindex[s[t]]] + np.roll(x,1))
        x = x/np.linalg.norm(x)
        X[:,t],S[:,t] = x,v[:,characterindex[s[t+1]]]
    XX = np.dot(X,X.T)
    for n in range(N):
        XX[n,n] = XX[n,n] + eta
    w = np.dot(np.dot(S,X.T),np.linalg.inv(XX))
    ss = recall(T,N,w,u,c,alpha,s[0])
    print("err=",error(s,ss),"%\n",ss,"\n")
    return ss,w

def online_learning(u,v,c,a,s):
    T,(N,M) = len(s),u.shape
    w,err,tt = np.zeros((M,N)),100.,0
    while err>0 and tt<T:
        x = np.zeros(N)
        #print(x)
        for t in range(T-1):
            x = (1.0-a)*x + a*(u[:,characterindex[s[t]]] + np.roll(x,1))
            # if t < 5:
            #     print(x)
            x = x/np.linalg.norm(x)
            # if t < 5:
            #     print(x)
            p = np.exp(np.dot(w,x))
            p = p/np.sum(p)
            w = w + np.outer(v[:,characterindex[s[t+1]]]-p,x)
        ss = recall(T,N,w,u,c,a,s[0])
        err,tt = error(s,ss),tt+1
        if tt % 60 == 0:
            print(tt,"err=",err,"%\n",ss,"\n")
    print(tt,"err=",err,"%\n",ss,"\n")
    return ss,w



s1 = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer--excellent for drawing the veil from men’s motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results. "
s2 = "Grit in a sensitive instrument, or a crack in one of his own high-power lenses, would not be more disturbing than a strong emotion in a nature such as his. And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory."
#s = s1 + s2
with open('test02.txt', 'r') as ofile:
    s = ofile.read()
n=2
# grams = cg.chargrams(s)
# glist = []
# for ngram, cnt in grams.items():
#     glist.append(ngram)
# list of all characters in text
# allchars = list(set(s))
# character index map with the form:
# 'H': 0, 'p': 3, 's': 38, etc.
# characterindex = {char:idx for idx,char in enumerate(allchars)}
# gramindex = {gram:idx for idx,gram in enumerate(glist)}
#print(gramindex)

glist = []
with open('counts1024.txt', 'r') as countfile:
    counts = countfile.readlines()
for line in counts:
    glist.append(line[:2])
gramindex = {gram:idx for idx,gram in enumerate(glist)}
print(len(gramindex))

sgi = []
sg = cg.gut_clean(s)
for idx in range(0, len(sg) - (n - 1), 2):
    sgi.append(gramindex[sg[idx:idx + n]])
print(len(sgi))

# T = length of sequence
# M = number of distinct input states
#T,M = len(s),len(allchars)
T,M = len(sgi),len(glist)

# mt = ratio of possible input states to timesteps to memorize
mt = M / T
# nt = ratio of reservoir units to timesteps to memorize

# N = number of hidden states/reservoir size (determined by ratio nt calculated above)
# alpha = integration rate indicating "leakiness", 1 = no leaking/orig model
# alpha is determined by ratio nt (hidden states) minus ratio mt (input states)
#N,alpha = int(0.5*T),0.5
#N = int(M * 1.25)
tnt = int(T * 0.62)
maxn = [tnt, M]
N = np.amax(maxn)

nt = N / T
nmt = [str(mt), str(nt)]
div = ' mt:nt '
print(div.join(nmt))
alpha = 0.58
print(alpha, N)
#N = int(0.5*T)
#alpha = 0.305099511

np.random.seed(11712)
# u = input weight matrix
# v = hidden state orthogonal identity matrix
u,v = init(M,N)

# ss,w = offline_learning(u,v,allchars,alpha,s)
# ss,w = online_learning(u,v,allchars,alpha,s)
totalo = time.perf_counter()
ss,w = offline_grams(u,v,glist,alpha,sgi)
endtotalo = time.perf_counter() - totalo
print(endtotalo)

#ss,w = online_grams(u,v,glist,alpha,sgi)
print(T,N,M,alpha)
