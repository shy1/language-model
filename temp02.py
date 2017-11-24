#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle
import re
from random import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import cupy as cp


outweights = '/home/user01/dev/language-model/outweights' + str(4096) + "-" + str(0) + ".p"
inweights = '/home/user01/dev/language-model/inweights' + str(4096) + "-" + str(0) + ".p"

inweights = '/home/user01/dev/language-model/inweights0.p0.p'
outweights = '/home/user01/dev/language-model/outweights0.p0' + ".p"

saved_outweights = pickle.load(open(outweights, "rb"))
saved_iweights = pickle.load(open(inweights, "rb"))
print(saved_iweights["U1"].shape, type(saved_iweights["U1"]))
print(saved_outweights["W1"].shape, type(saved_outweights["W1"]))
print(saved_iweights)
print(saved_outweights)
chunkfile = '/home/user01/dev/language-model/chunks100mb.p'
outweights = '/home/user01/dev/language-model/outweights100mb.p'
inweights = '/home/user01/dev/language-model/inweights100mb.p'
#pickle.dump(allchunks, open(outfile, "wb"))
chunklist = pickle.load(open(chunkfile, "rb"))
print(len(chunklist))
intchunklist = []
n = 2
stride = 1
wv = KeyedVectors.load_word2vec_format('/home/user01/dev/wang2vec/embeddings-i3e4-ssg-neg15-s1024w6.txt', binary=False)
temp = wv.index2word
glist = np.array(temp[1:len(temp)])
gramindex = {gram:idx for idx, gram in enumerate(glist)}
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

for i in range(3):

    print(intchunklist[3])

    shuffle(intchunklist)

def findFiles(path): return glob.glob(path)

inputpath='/home/user01/dev/data/gutenberg/nochapters-nl'
inputpath='/home/user01/dev/data/gutenberg/nochapters-nl/test'
inputpath = inputpath + '/*.txt'
allchunks = []


for filename in findFiles(inputpath):
    with open(filename, "r") as f:
        raw_text = f.read()

    raw_text = raw_text.strip()
    splits = raw_text.split(". ")
    chunk = ""

    for i in range(len(splits)):        
        chunk += " " + splits[i] + "."
        if len(chunk) > 256:
        # pad with space if chunk has an odd number of characters so that
        # chunk length is compatibile with bigrams using a stride of 2
            if len(chunk) % 2 != 0:
                chunk += " "
            allchunks.append(chunk)
            chunk = ""
    print(len(allchunks))

for filename in findFiles(inputpath):
    with open(filename, "r") as f:
        raw_text = f.read()

    raw_text = raw_text.strip()
    splits = raw_text.split(' ')
    chunk = ''

    for i in range(len(splits)):        
        chunk += ' ' + splits[i]
        if len(chunk) >= 136:
            chunk = chunk[0:128]
            allchunks.append(chunk)

            chunk = ''
    print(len(allchunks))
print(allchunks[24:36])
#print(len(allchunks[4]), allchunks[4])
#print(len(allchunks[9]), allchunks[9])
print(len(allchunks[1700]), allchunks[1700])
shuffle(allchunks)
print('\nchunk [1700] after shuffling')
print(len(allchunks[1700]), allchunks[1700])

outfile = '/home/user01/dev/language-model/chunks136.p'

pickle.dump(allchunks, open(outfile, "wb"))

filechunks = pickle.load(open(outfile, "rb"))
glist = np.array(temp[1:len(temp)])
glist = [re.sub(r'_', ' ', j) for j in glist]
gramindex = {gram:idx for idx, gram in enumerate(glist)}

chars = 0
trainchunks = []
testchunks = []

for j in range(2800):
    chunk = filechunks[j]
    sgi = []
    for idx in range(0, len(chunk) - (n - 1), stride):
        try:
            sgi.append(gramindex[chunk[idx:idx + n]])
        except:
            print(chunk[idx:idx + n])
    intchunk = np.asarray(sgi, dtype=np.int16)
    chars += len(intchunk)
    trainchunks.append(intchunk)
print(chars)

chars = 0
for j in range(2700, 3000):
    chunk = filechunks[j]
    sgi = []
    for idx in range(0, len(chunk) - (n - 1), stride):
        try:
            sgi.append(gramindex[chunk[idx:idx + n]])
        except:
            print(chunk[idx:idx + n])
    intchunk = np.asarray(sgi, dtype=np.int16)
    chars += len(intchunk)
    testchunks.append(intchunk)
print(chars)
print(len(trainchunks), len(testchunks))

trainfile = '/home/user01/dev/language-model/train1m.p'
testfile = '/home/user01/dev/language-model/test1m.p'
pickle.dump(trainchunks, open(trainfile, "wb"))
pickle.dump(testchunks, open(testfile, "wb"))
trainchunks = pickle.load(open(trainfile, "rb"))
print(trainchunks[0])
#for i in range(len(filechunks)):
#    filechunks[i] = re.sub('*', ' ', filechunks[i])
#print('\nchunk [1700] after loading from pickled file')
#print(len(filechunks[1700]), filechunks[1700])

#parsedData = parser(raw_text)
#
#sents = []
#
#for span in parsedData.sents:
#    # go from the start to the end of each span, returning each token in the sentence
#    # combine each token using join()
#    sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
#    sents.append(sent)
#
#for i in range(50, 80):
#    print(i, len(sents[i]), sents[i])
#
#print(len(sents))