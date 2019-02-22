import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import math
import nltk
from nltk import word_tokenize
from nltk.corpus import reuters
import pickle

with open('vocab.txt') as vocab_file:
    lines = vocab_file.readlines()
vocab = [line.strip() for line in lines]

testdata = pd.read_table('testdata.txt', header=None)
n = testdata.shape[0]

with open('dataFile','rb') as fr:
    un_freq = pickle.load(fr)
    bi_freq = pickle.load(fr)
    V = pickle.load(fr)
    N = pickle.load(fr)

#un_freq = nltk.FreqDist(reuters.words())
#bigrams = nltk.bigrams(reuters.words())
#bi_freq = nltk.FreqDist(bigrams)
#V = len(un_freq)
#N = len(reuters_list)

letters = "abcdefghijklmnopqrstuvwxyz"
letters_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
other = " -'"
def find_upper(word):
    count = 0
    pos = []
    for i in range(0,len(word)):
        if word[i].isupper():
            count += 1
            pos.append(i)
    return [count,pos]

def uppercase_corr(word):
    total = {word}
     #UPPERCASES
    find = find_upper(word)
    num = find[0]
    ind = find[1]
    if num == 1 and ind[0] == 0:
        return {word}
    elif num == 1 and ind[0] > 0: #One uppercase
        new_word = word[ind[0]]+word[1:ind[0]]+word[0]+word[ind[0]+1:]
        return {new_word,word}
    elif num == len(word)-1: #e.g. INTERNAsIONAL
        for i in range(0,len(word)):
            if i not in ind:
                l = word[:i]
                r = word[i:]
                alphabet = letters_upper
                if len(word) > 1 and i > 0:
                    if i < len(word):
                        alphabet = letters_upper+other # Punctuation cases
                total.update(set([l+r[1:]])) #Deletion
                total.update(set([l+c+r[1:] for c in alphabet])) #Substitution
                break
        return total
    else: #e.g. INTERVNTION ltGR
        for i in range(0,len(word)):
            l = word[:i]
            r = word[i:]
            alphabet = letters_upper
            if len(word) > 1 and i > 0:
                #total.update(set([l[:i-1]+r[0]+l[i-1]+r[1:]])) #Transposition
                if i < len(word):
                    alphabet = letters_upper+other # Punctuation cases
            total.update(set([l+c+r for c in alphabet]+[l+r+c for c in alphabet])) #Insertion
            #total.update(set([l+r[1:]])) #Deletion
            total.update(set([l+c+r[1:] for c in alphabet])) #Substitution
        return total
    return total

def distance_1(word):
    total = {word}
    for i in range(0,len(word)):
        l = word[:i]
        r = word[i:]
        alphabet = letters
        if len(word) > 1 and i > 0:
            total.update(set([l[:i-1]+r[0]+l[i-1]+r[1:]])) #Transposition
            if i < len(word):
                alphabet = letters+other # Punctuation cases
        total.update(set([l+c+r for c in alphabet]+[l+r+c for c in alphabet])) #Insertion
        total.update(set([l+r[1:]])) #Deletion
        total.update(set([l+c+r[1:] for c in alphabet])) #Substitution
    total = total - {""}
    return total

def non_distance_1(word):
    total = set()
    for i in range(0,len(word)-1):
        for j in range(i+1,len(word)):
            if word[i] != word[j]:
                new_word = word[:i]+word[j]+word[i+1:j]+word[i]+word[j+1:]
                total.add(new_word)
    return total

def log_smoothed_prob(pre,next):
    prob = math.log10((bi_freq[(pre,next)]+1))-math.log10((un_freq[pre]+V))
    return prob

def correction(pre,word,vocab):
    prob_dict = dict()

    find = find_upper(word)
    num = find[0]
    if num > 0:
        for y in uppercase_corr(word):
            if y in vocab:
                prob_dict[y]=log_smoothed_prob(pre,y)
    for y in distance_1(word):
        if y in vocab:
            prob_dict[y]=log_smoothed_prob(pre,y)
    for y in non_distance_1(word):
            if y in vocab:
                prob_dict[y]=log_smoothed_prob(pre,y)
    if len(prob_dict) == 0:
        return word
    else:
        return max(prob_dict,key=prob_dict.get)

def generate_candidates(sentence):
    candidates = list([sentence])
    for p,word in enumerate(sentence):
        if word[0] not in letters+letters_upper:
            continue
        elif word[0] in letters_upper: # initial letter uppercase
            for replace in distance_1(word.lower()):
                if len(replace) == 1:
                    replace = replace[0].upper()
                else:
                    replace = replace[0].upper()+replace[1:]
                if replace in vocab:
                    candidate = sentence[:p]+[replace]+sentence[p+1:]
                    if candidate not in candidates:
                        candidates.append(candidate)
        else:
            for replace in distance_1(word):
                if replace in vocab:
                    candidate = sentence[:p]+[replace]+sentence[p+1:]
                    if candidate not in candidates:
                        candidates.append(candidate)
    return candidates

def sentence_prob(sentence):
    log_prob = 0
    for p,word in enumerate(sentence):
        if p == 0:
            log_prob = math.log10(un_freq[word]+1) - math.log10(N+V)
        else:
            pre = sentence[p-1]
            new_prob = log_smoothed_prob(pre,word)
            log_prob = log_prob + new_prob
    return log_prob

def real_word_correction(sentence):
    max_prob = -10000
    best_candidate = sentence
    for candidate in generate_candidates(sentence):
        prob = sentence_prob(candidate)
        if prob > max_prob:
            max_prob = prob
            best_candidate = candidate
    return best_candidate

exist_real_word_errors = list()
result = testdata.drop(columns=1)
for i in range(0,n):
    non_word_count = 0
    sentence = word_tokenize(testdata[2][i])
    for p, word in enumerate(sentence):
        if non_word_count == testdata[1][i]:
           break
        if word not in vocab:
            non_word_count += 1
            #correct_word = correction(sentence[p-1],word,vocab)
            #print(str(i+1)+" "+word+" "+correct_word)
            #result.iat[i,1] = result.iat[i,1].replace(word,correct_word)
    if non_word_count != testdata[1][i]:
        exist_real_word_errors.append(i)

#for i in exist_real_word_errors:
#    print(i)

for i in exist_real_word_errors:
    if i<498:
        continue
    sentence = word_tokenize(testdata[2][i])
    correct_sentence = real_word_correction(sentence)
    for j in range(0,len(sentence)):
        if sentence[j] != correct_sentence[j]:
            word = sentence[j]
            correct_word = correct_sentence[j]
            result.iat[j,1] = result.iat[j,1].replace(word,correct_word)
            break
    print(str(i+1)+" "+word+" "+correct_word)


#np.savetxt('result.txt',result.values,fmt='%s',delimiter='\t',)