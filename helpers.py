# https://github.com/spro/practical-pytorch

import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable
import re

# Reading and un-unicode-encoding data

all_characters = string.ascii_lowercase + string.digits + string.punctuation + " "
n_characters = len(all_characters)

def read_file(filename):
    #file = unidecode.unidecode(open(filename).read())
    #file = gut_clean(file)
    file = open(filename).read()
    return file, len(file)


# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def removeHardwraps(book):
    # raw gutenberg data contains extraneous newline characters '\n' for word wrapping
    # to specific column lengths
    # 1. replace true paragraph segmentations '\n\n' with a placeholder 'ppQQpp'
    # 2. convert all false newlines to the correct character which is a space
    # 3. return the correct double newlines back into their placeholder spots
    book = re.sub('\n\n', 'ppQQpp', book)
    book = re.sub('\n', ' ', book)
    book = re.sub('ppQQpp', '\n\n', book)
    return book

def gut_clean(book):
    book = removeHardwraps(book)
    clean_input = re.compile(r'[\t\n]').sub(' ', book)
    # replace sequences of two or more spaces with a single space
    clean_input = re.sub(' {2,}', ' ', clean_input)
    # convert all letters to lower case
    clean_input = clean_input.lower()
    for c in clean_input:
        if c not in all_characters:
            clean_input = re.sub(c, '#', clean_input)
    return clean_input
