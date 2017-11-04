#import chargrams as cg
import glob
import os
import re

def findFiles(path): return glob.glob(path)
def openbook(filename):
    with open(filename, 'r') as ofile:
        book = ofile.read()
    return book

def write2single(cgrams, filename):
    book = ''
    space  = ' '
    book = space.join(cgrams)

    with open(filename, 'a') as ofile:
        ofile.write(book)
        ofile.write("\n")

def writecgrams(cgrams, filename):
    book = ''
    space  = ' '
    book = space.join(cgrams)

    with open(filename, 'w') as ofile:
        ofile.write(book)

def writebigrams(bigramlist, filename):
    book = ''
    space  = ' '
    book = space.join(bigramlist)

    with open(filename, 'w') as ofile:
        ofile.write(book)

def getgramindex(filename='counts1024b.txt'):
    glist = []
    with open(filename, 'r') as countfile:
        counts = countfile.readlines()
    for line in counts:
        glist.append(line[:2])
    gramindex = {gram:idx for idx,gram in enumerate(glist)}
    print(len(gramindex))
    return gramindex

def replaceasterisk(sgram):
    temp1 = sgram[0] + '*'
    temp2 = '*' + sgram[1]
    print(temp1, temp2)

def replacespace(sgram):
    temp1 = sgram[0] + '_'
    temp2 = '_' + sgram[1]
    #print(temp1, temp2)
    return temp1, temp2

def convert2chargrams(filename, gramindex):
    n=2
    book = openbook(filename)
    book = re.compile(r'\t').sub(' ', book)
    #book = re.sub(' \n', '\n', book)
    #book = re.sub('\n ', '\n', book)
    #book = re.sub('\n', ' \n', book)
    book = re.sub('\n', ' \n ', book)
    #book = re.sub('_\n', '\n', book)
    #book = re.sub('\n_', '\n', book)
    book = re.sub(' {2,}', ' ', book)
    cgrams = []
    #sgrams = []
    for idx in range(0, len(book) - (n - 1), 2):
        cgram = book[idx:idx + n]
        #print(cgram)
        #sgrams.append(sgram)
        if cgram in gramindex:
            cgram = re.sub(' ', '_', cgram)
            cgrams.append(cgram)
        elif "\n" not in cgram:
            temp1, temp2 = replacespace(cgram)
            cgrams.extend([temp1, temp2])
        else:
            #cgram = "\n"
            cgrams.append(cgram)
    return cgrams

def convert2idx(filename, gramindex):
    n=2
    book = openbook(filename)
    bigrams = []
    #sgrams = []
    for idx in range(0, len(book) - (n - 1)):
        sgram = book[idx:idx + n]
        #sgrams.append(sgram)
        if sgram in gramindex:
            bigrams.append(str(gramindex[sgram]))
        else:
            temp1, temp2 = replacespace(sgram)
            bigrams.extend([str(gramindex[temp1]), str(gramindex[temp2])])
    return bigrams

def folder2bigrams(inputpath='/home/user01/dev/data/gutenberg/nolines', outputpath='/home/user01/dev/data/gutenberg/bigrams/', n=2):
    inputpath = inputpath + '/*.txt'

    gramindex = getgramindex()
    i = 0
    for filename in findFiles(inputpath):
        i += 1
        bigrams = convert2idx(filename, gramindex)

        temp = os.path.split(filename)
        outputbook = outputpath + temp[1]
        writebigrams(bigrams, outputbook)
        print(i)

def folder2cgrams(inputpath='/home/user01/dev/data/gutenberg/cleaner', outputpath='/home/user01/dev/data/gutenberg/chargrams/', n=2):
    inputpath = inputpath + '/*.txt'

    gramindex = getgramindex()
    i = 0
    for filename in findFiles(inputpath):
        i += 1
        cgrams = convert2chargrams(filename, gramindex)

        #temp = os.path.split(filename)
        #outputbook = outputpath + temp[1]
        #writecgrams(cgrams, outputbook)
        
        write2single(cgrams, '/home/user01/dev/data/gutenberg/chargrams/concatenated-w2.txt')
        print(i)

#folder2bigrams()
folder2cgrams()
# bigrams = convert2idx('/home/user01/dev/data/gutenberg/nolines/Joseph Conrad___The Secret Sharer.txt')
# print(len(bigrams), type(bigrams))
# writebigrams(bigrams, 'bigtest.txt')
# writesgrams(sgrams, 'stringtest.txt')
