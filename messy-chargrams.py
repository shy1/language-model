import argparse
import collections
import re
import sys
from operator import itemgetter
import string
import unidecode
import glob
import os
import cchardet as chardet

all_characters = string.ascii_lowercase + " 9!\"$\'()*,-./:;?^\n\t"
n_characters = len(all_characters)
count = collections.defaultdict(int)
outputlist = ''

def chargrams(book, n=2):
    # add a space character to the beginning of full text input so that
    # the first chargram is recognized as the start of a word
    #book = " " + book
    # create empty dictionary (chargram + integer frequency count) and empty outputlist string
    count = collections.defaultdict(int)
    outputlist = ''
    # replace tabs and newlines with spaces
    #clean_input = re.compile(r'[\t\n]').sub(' ', book)
    # replace sequences of two or more spaces with a single space
    #clean_input = re.sub(' {2,}', ' ', clean_input)
    # convert all letters to lower case
    #clean_input = clean_input.lower()
    for idx in range(0, len(book) - (n - 1)):
        count[book[idx:idx + n]] += 1
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

def separate1(text, char):
    qsplit = text.split(char)
    s = '^' + char
    text = s.join(qsplit)
    return text

def separate2(text, char):
    qsplit = text.split(char)
    s = char + '^'
    text = s.join(qsplit)
    return text

def substitute(text, c1, c2):
    ssplit = text.split(c1)
    s = c2
    text = s.join(ssplit)
    return text

def gut_clean(book):
    book = removeHardwraps(book)
    #clean_input = re.compile(r'[\t\n]').sub(' ', book)
    # replace sequences of two or more spaces with a single space
    clean_input = re.sub(' {2,}', ' ', book)
    # convert all letters to lower case
    clean_input = clean_input.lower()
    prev_c = " "
    both_c = " "
    clean_input = re.sub('_', '', clean_input)
    clean_input = re.sub('[]>}]', ')', clean_input)
    clean_input = re.sub('[[{<]', '(', clean_input)

    for char in "!$(),.:;?/*":
        clean_input = separate1(clean_input, char)
    for char in "!$(),.:;?/*":
        clean_input = separate2(clean_input, char)

    clean_input = re.sub('[`]', '\'', clean_input)
    clean_input = re.sub('[\\\\]', '/', clean_input)
    clean_input = re.sub('[0-8]', '9', clean_input)
    clean_input = re.sub('["]', '^\"^', clean_input)
    clean_input = re.sub('\^{2,}', '^', clean_input)
    clean_input = re.sub('\^ | \^', ' ', clean_input)

    for c in clean_input:
        try:
            if c not in all_characters:
                clean_input = substitute(clean_input, c, '*')
                # if c in '012345678':
                    #clean_input = re.sub(c, '9', clean_input)
                #     a = 1
                # elif c == '`':
                #     clean_input = re.sub(c, '\'', clean_input)
                # elif c == '\\':
                #     c = '\\\\'
                #     clean_input = re.sub(c, '/', clean_input)
                # else:
                #     #clean_input = re.sub(c, '_', clean_input)

            # if prev_c == "*":
            #     both_c = "\\" + prev_c + c
            #     sepa_c = prev_c + '^' + c
            #     if (c != " "):
            #         clean_input = re.sub(both_c, sepa_c, clean_input)
        except:
            print(c)
        #prev_c = c
    clean_input = re.sub('[*]', '^*^', clean_input)

    #clean_input = re.sub('(\w)\*', '\1^*', clean_input)

    # qsplit = clean_input.split('^ ')
    # s = ' '
    # clean_input = s.join(qsplit)


        #     else:
        #         both_c = prev_c + c
        #     if c in "$()*+./?[]|":
        #         both_c = prev_c + "\\" + c
        #     else:
        #         sepa_c = prev_c + '^' + c
        #     if (c in "!\"#$&\()*+,-./:;=?[]_|~\n\t") and (prev_c != " "):
        #         clean_input = re.sub(both_c, sepa_c, clean_input)
            # if (prev_c in "!\"#$&()*+,-./:;=?[]_|~\n\t") and (c != " "):
            #     clean_input = re.sub(both_c, sepa_c, clean_input)
    # for char in "/*":
    #     clean_input = separate1(clean_input, char)
    # for char in "/*":
    #     clean_input = separate2(clean_input, char)


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

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    file = gut_clean(file)
    return file

def findFiles(path): return glob.glob(path)

def cdetect(inputpath):
    outputlist = ''

    for filename in findFiles(inputpath):
        with open(filename, "rb") as f:
            msg = f.read()
            result = chardet.detect(msg)
            outputlist = outputlist + '\n' + str(result['encoding']) + ' - ' + str(filename)

    with open('encodings.txt', 'w') as ofile:
        ofile.write(outputlist)

def folder2cgrams(inputpath, outputfile='counts4.txt', n=2):
    count = collections.defaultdict(int)
    outputlist = ''
    outputpath = inputpath + '/separated/'
    inputpath = inputpath + '/*.txt'
    ## section for cleaning texts/books if required
    # for filename in findFiles(inputpath):
    #     with open(filename, "rb") as f:
    #         msg = f.read()
    #         result = chardet.detect(msg)
    #     if (result['encoding'] == 'UTF-8') or (result['encoding'] == 'ASCII'):
    #         book = read_file(filename)
    #         book = removeHardwraps(book)
    #         book = re.sub(' {2,}', ' ', book)
    #         # convert all letters to lower case
    #         book = book.lower()
    for filename in findFiles(inputpath):
        with open(filename, "r") as f:
            book = f.read()
        book = gut_clean(book)
        for idx in range(0, len(book) - (n - 1)):
            count[book[idx:idx + n]] += 1
        temp = os.path.split(filename)
        outputbook = outputpath + temp[1]
        with open(outputbook, 'w') as ofile:
            ofile.write(book)
    for ngram, cnt in reversed(sorted(count.items(), key=itemgetter(1))):
        listitem = u'{}\t{}'.format(ngram, cnt)
        #print(listitem)
        outputlist = outputlist + '\n' + listitem
    print(len(count))

    #print(outputlist)
    outputlist = outputlist[1:]

    with open(outputfile, 'w') as ofile:
        ofile.write(outputlist)

def file2cgrams(inputfile, outputfile, gramlength=1):
    with open(inputfile, 'r') as ifile:
        filetext = ifile.read()
    filetext = removeHardwraps(filetext)
    filetext = gut_clean(filetext)
    outputlist = chargrams(filetext, gramlength)
    with open(outputfile, 'w') as ofile:
        ofile.write(outputlist)

def cleanFolder(inputpath):
    for filename in findFiles(inputpath):
        with open(filename, "rb") as f:
            msg = f.read()
            result = chardet.detect(msg)
        if (result['encoding'] == 'UTF-8') or (result['encoding'] == 'ASCII'):
            book = read_file(filename)
            temp = os.path.split(filename)
            outputfile = temp[0] + "/cleaned/" + temp[1]
            print(outputfile)
            with open(outputfile, 'w') as ofile:
                ofile.write(book)
