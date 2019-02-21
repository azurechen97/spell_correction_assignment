import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from nltk import word_tokenize
from nltk.corpus import reuters

letters = "abcdefghijklmnopqrstuvwxyz"
other = " -'"
def distance_1(word):
    total = {word}
    for i in range(0,len(word)):
        l = word[:i]
        r = word[i:]
        alphabet = letters
        if len(word) > 1 and i > 0:
            total.update(set([l[:i-1]+r[0]+l[i-1]+r[1:]])) #Transposition
            if i < len(word):
                alphabet = letters+other
        total.update(set([l+c+r for c in alphabet]+[l+r+c for c in alphabet])) #Insertion
        total.update(set([l+r[1:]])) #Deletion
        total.update(set([l+c+r[1:] for c in alphabet])) #Substitution
    return total

def distance_2(word):
    return set(d2 for d1 in distance_1(word) for d2 in distance_1(d1))

def correction(word,vocab):
    for y in distance_1(word):
        if y in vocab:
            return y
    for y in distance_1(word.lower()):
        if y in vocab:
            return y
    return "Not Found"

with open('vocab.txt') as vocab_file:
    lines = vocab_file.readlines()
vocab = [line.strip() for line in lines]

testdata = pd.read_table('testdata.txt', header=None)
n = testdata.shape[0]

exist_real_word_errors = list()
for i in range(0,n):
    misspell_count = 0
    sentence = word_tokenize(testdata[2][i])
    for p, word in enumerate(sentence):
        if misspell_count == testdata[1][i]:
           break
        if word not in vocab:
            misspell_count += 1
            print(word+" "+correction(word,vocab))
    if misspell_count != testdata[1][i]:
        exist_real_word_errors.append([i, testdata[1][i] - misspell_count])