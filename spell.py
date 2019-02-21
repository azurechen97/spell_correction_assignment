import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import nltk

letters = "abcdefghijklmnopqrstuvwxyz"
other = " -'"
def distance_1(word):
    total = {word}
    for i in range(0,len(word)):
        l = word[:i]
        r = word[i:]
        alphabet = letters
        if len(word) > 1 and i > 0:
            total.update(set([l[:i-1]+r[0]+l[i-1]+r[1:]]))
            if i < len(word):
                alphabet = letters+other
        total.update(set([l+c+r for c in alphabet]))
        total.update(set([l+r[1:]]))
        total.update(set([l+c+r[1:] for c in alphabet]))
    return total

def correction(word):
    return word

with open('vocab.txt') as vocab_file:
    lines = vocab_file.readlines()
vocab = [line.strip() for line in lines]

testdata = pd.read_table('testdata.txt', header=None)
n = testdata.shape[0]

exist_real_word_errors = list()
for i in range(0,20):
    misspell_count = 0
    sentence = nltk.word_tokenize(testdata[2][i])
    for p, word in enumerate(sentence):
        if misspell_count == testdata[1][i]:
           break
        if word not in vocab:
            misspell_count += 1
            
    if misspell_count != testdata[1][i]:
        exist_real_word_errors.append([i, testdata[1][i] - misspell_count])