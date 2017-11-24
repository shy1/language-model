#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 00:41:26 2017

@author: user01
"""
import pickle
import re

outfile = '/home/user01/dev/language-model/chunks100mb.p'

filechunks = pickle.load(open(outfile, "rb"))
newchunks = []
for i in range(len(filechunks)):
    tempchunk = re.sub(r'\*i', ' i', filechunks[i])
    tempchunk = re.sub(' {2,}', ' ', tempchunk)
    newchunks.append(tempchunk)
pickle.dump(newchunks, open(outfile, "wb"))
