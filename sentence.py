import math
import pickle
from nltk import word_tokenize

with open('vocab.txt') as vocab_file:
    lines = vocab_file.readlines()
vocab = [line.strip() for line in lines]

with open('dataFile','rb') as fr:
    un_freq = pickle.load(fr)
    bi_freq = pickle.load(fr)
    V = pickle.load(fr)
    N = pickle.load(fr)

letters = "abcdefghijklmnopqrstuvwxyz"
letters_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
                alphabet = letters+other # Punctuation cases
        total.update(set([l+c+r for c in alphabet]+[l+r+c for c in alphabet])) #Insertion
        total.update(set([l+r[1:]])) #Deletion
        total.update(set([l+c+r[1:] for c in alphabet])) #Substitution
    return total

def generate_candidates(sentence):
    candidates = list()
    for p,word in enumerate(sentence):
        for replace in distance_1(word):
            if replace in vocab:
                candidate = sentence[:p]+[replace]+sentence[p+1:]
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates

def log_smoothed_prob(pre,next):
    prob = math.log10((bi_freq[(pre,next)]+1))-math.log10((un_freq[pre]+V))
    return prob

def sentence_prob(sentence):
    log_prob = 0
    for p,word in enumerate(sentence):
        if p == 0:
            log_prob = 0
            #math.log10(un_freq[word]+1) - math.log10(N+V)
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

word = input("input: ")
print(real_word_correction(word_tokenize(word)))