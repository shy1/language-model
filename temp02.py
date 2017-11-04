#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 01:29:36 2017

@author: user01
"""

from spacy.en import English
parser = English()

file_name = "/home/user01/dev/data/gutenberg/cleaner/Anthony Trollope___Kept in the Dark.txt"
with open(file_name) as f:
    raw_text = f.read()

parsedData = parser(raw_text)

sents = []

for span in parsedData.sents:
    # go from the start to the end of each span, returning each token in the sentence
    # combine each token using join()
    sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
    sents.append(sent)

for i in range(50, 80):
    print(i, len(sents[i]), sents[i])

print(len(sents))