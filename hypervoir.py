# Reservoir computing on the hypersphere
import numpy as np

# create scaling input matrix U and hidden state reservoir matrix V
def init(M,N):
    u,v = np.random.rand(N,M),np.identity(M)
    # normalizing columns in NxM sized input matrix U as in formulas 6,7
    for m in range(M):
        u[:,m] = u[:,m] - u[:,m].mean()
        u[:,m] = u[:,m]/np.linalg.norm(u[:,m])
    return u,v

def recall(T,N,w,u,c,a,ss):
    x,i = np.zeros(N),ci[ss]
    for t in range(T-1):
        x = (1.0-a)*x + a*(u[:,i] + np.roll(x,1))
        x = x/np.linalg.norm(x)
        y = np.exp(np.dot(w,x))
        i = np.argmax(y/np.sum(y))
        ss = ss + str(c[i])
    return ss

def error(s,ss):
    err = 0.
    for t in range(len(s)):
        err = err + (s[t]!=ss[t])
    return np.round(err*100.0/len(s),2)

def offline_learning(u,v,c,a,s):
    T,(N,M),eta = len(s),u.shape,1e-7
    X,S,x = np.zeros((N,T-1)),np.zeros((M,T-1)),np.zeros(N)
    for t in range(T-1):
        x = (1.0-a)*x + a*(u[:,ci[s[t]]] + np.roll(x,1))
        x = x/np.linalg.norm(x)
        X[:,t],S[:,t] = x,v[:,ci[s[t+1]]]
    XX = np.dot(X,X.T)
    for n in range(N):
        XX[n,n] = XX[n,n] + eta
    w = np.dot(np.dot(S,X.T),np.linalg.inv(XX))
    ss = recall(T,N,w,u,c,alpha,s[0])
    print "err=",error(s,ss),"%\n",ss,"\n"
    return ss,w

def online_learning(u,v,c,a,s):
    T,(N,M) = len(s),u.shape
    w,err,tt = np.zeros((M,N)),100.,0
    while err>0 and tt<T:
        x = np.zeros(N)
        for t in range(T-1):
            x = (1.0-a)*x + a*(u[:,ci[s[t]]] + np.roll(x,1))
            x = x/np.linalg.norm(x)
            p = np.exp(np.dot(w,x))
            p = p/np.sum(p)
            w = w + np.outer(v[:,ci[s[t+1]]]-p,x)
        ss = recall(T,N,w,u,c,a,s[0])
        err,tt = error(s,ss),tt+1
        print tt,"err=",err,"%\n",ss,"\n"
    return ss,w

s = \
"To Sherlock Holmes she is always THE woman. I have seldom heard him mention \
her any other name. In his eyes she eclipses and predominates the whole of \
her sex. It was not that he felt any emotion akin to love for Irene Adler. \
All emotions, and that one particularly, were abhorrent to his cold, precise \
but admirably balanced mind. He was, I take it, the most perfect reasoning \
and observing machine that the world has seen, but as a lover he would have \
placed himself in a false position. He never spoke of the softer passions, \
save with a gibe and a sneer. They were admirable things for the observer--\
11 excellent for drawing the veil from men’s motives and actions. But for the \
trained reasoner to admit such intrusions into his own delicate and finely \
adjusted temperament was to introduce a distracting factor which might throw \
a doubt upon all his mental results. Grit in a sensitive instrument, or a crack \
in one of his own high-power lenses, would not be more disturbing than a strong \
emotion in a nature such as his. And yet there was but one woman to him, and \
that woman was the late Irene Adler, of dubious and questionable memory."

# list of all characters in text
c = list(set(s))
# character index
ci = {ch:m for m,ch in enumerate(c)}
# T = length of sequence
# M = number of distinct input states
T,M = len(s),len(c)
# N = number of hidden states
N,alpha = int(0.5*T),0.5

np.random.seed(12345)
u,v = init(M,N)

ss,w = offline_learning(u,v,c,alpha,s)
ss,w = online_learning(u,v,c,alpha,s)

print T,N,M,alpha
