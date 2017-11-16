#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle
import re
from random import shuffle



def findFiles(path): return glob.glob(path)

inputpath='/home/user01/dev/data/gutenberg/nochapters-nl'
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
        if len(chunk) > 512:
#            print(len(chunk))
            allchunks.append(chunk)
            chunk = ""
    print(len(allchunks))

#print(len(allchunks[4]), allchunks[4])
#print(len(allchunks[9]), allchunks[9])
print(len(allchunks[1700]), allchunks[1700])
shuffle(allchunks)
print('\nchunk [1700] after shuffling')
print(len(allchunks[1700]), allchunks[1700])

outfile = '/home/user01/dev/language-model/chunks100mb.p'

pickle.dump(allchunks, open(outfile, "wb"))

filechunks = pickle.load(open(outfile, "rb"))

for i in range(len(filechunks)):
    filechunks[i] = re.sub('*', ' ', filechunks[i])
print('\nchunk [1700] after loading from pickled file')
print(len(filechunks[1700]), filechunks[1700])

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