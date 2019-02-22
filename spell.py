import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import math
import nltk
from nltk import word_tokenize
from nltk.corpus import reuters

with open('vocab.txt') as vocab_file:
    lines = vocab_file.readlines()
vocab = [line.strip() for line in lines]

testdata = pd.read_table('testdata.txt', header=None)
n = testdata.shape[0]

un_freq = nltk.FreqDist(reuters.words())
V = len(un_freq)
bigrams = nltk.bigrams(reuters.words())
bi_freq = nltk.FreqDist(bigrams)

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
    return set(d2 for d1 in distance_1(word) for d2 in distance_1(d1)) - distance_1(word)

def log_smoothed_prob(pre,next):
    prob = math.log10((bi_freq[(pre,next)]+1))-math.log10((un_freq[pre]+V))
    return prob

def correction(pre,word,vocab):
    prob_dict = dict()
    for y in distance_1(word):
        if y in vocab:
            prob_dict[y]=log_smoothed_prob(pre,y)
    if len(prob_dict) == 0:
        return "Not Found"
    else:
        return max(prob_dict,key=prob_dict.get)
    
exist_real_word_errors = list()
for i in range(0,100):
    misspell_count = 0
    sentence = word_tokenize(testdata[2][i])
    for p, word in enumerate(sentence):
        if misspell_count == testdata[1][i]:
           break
        if word not in vocab:
            misspell_count += 1
            print(word+" "+correction(sentence[p-1],word,vocab))
    if misspell_count != testdata[1][i]:
        exist_real_word_errors.append([i, testdata[1][i] - misspell_count])