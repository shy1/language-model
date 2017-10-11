import argparse
import collections
import re
import sys
from operator import itemgetter
import string

all_characters = string.ascii_lowercase + string.digits + string.punctuation + " "
n_characters = len(all_characters)

def chargrams(book, n=1):
    # add a space character to the beginning of full text input so that
    # the first chargram is recognized as the start of a word
    book = " " + book
    # create empty dictionary (chargram + integer frequency count) and empty outputlist string
    count = collections.defaultdict(int)
    outputlist = ''
    # replace tabs and newlines with spaces
    clean_input = re.compile(r'[\t\n]').sub(' ', book)
    # replace sequences of two or more spaces with a single space
    clean_input = re.sub(' {2,}', ' ', clean_input)
    # convert all letters to lower case
    clean_input = clean_input.lower()
    for idx in range(0, len(clean_input) - n):
        count[clean_input[idx:idx + n]] += 1
    for ngram, cnt in reversed(sorted(count.items(), key=itemgetter(1))):
        listitem = u'{}\t{}'.format(ngram, cnt)
        #print(listitem)
        outputlist = outputlist + '\n' + listitem
    print(len(count))
    outputlist = outputlist[1:]
    return outputlist

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

def clean_text(book):
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

def num2char(text1):
    text1 = re.sub(r"0", 'zero ', text1)
    text1 = re.sub(r"1", 'one ', text1)
    text1 = re.sub(r"2", 'two ', text1)
    text1 = re.sub(r"3", 'three ', text1)
    text1 = re.sub(r"4", 'four ', text1)
    text1 = re.sub(r"5", 'five ', text1)
    text1 = re.sub(r"6", 'six ', text1)
    text1 = re.sub(r"7", 'seven ', text1)
    text1 = re.sub(r"8", 'eight ', text1)
    text1 = re.sub(r"9", 'nine ', text1)
    return text1

def file2cgrams(inputfile, outputfile, gramlength=3):
    with open(inputfile, 'r') as ifile:
        filetext = ifile.read()
    filetext = removeHardwraps(filetext)
    filetext = clean_text(filetext)
    outputlist = chargrams(filetext, gramlength)
    with open(outputfile, 'w') as ofile:
        ofile.write(outputlist)
