import numpy as np
import pandas as pd
import nltk
from nltk.corpus import reuters
import pickle

reuters_list = reuters.words()
un_freq = nltk.FreqDist(reuters_list)
bigrams = nltk.bigrams(reuters_list)
bi_freq = nltk.FreqDist(bigrams)
V = len(un_freq)
N = len(reuters_list)

with open('dataFile','wb') as fw:
    pickle.dump(un_freq,fw)
    pickle.dump(bi_freq,fw)
    pickle.dump(V,fw)
    pickle.dump(N,fw)