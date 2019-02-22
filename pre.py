import numpy as np
import pandas as pd
import nltk
from nltk.corpus import reuters
import pickle

un_freq = nltk.FreqDist(reuters.words())
V = len(un_freq)
bigrams = nltk.bigrams(reuters.words())
bi_freq = nltk.FreqDist(bigrams)

fw = open('dataFile','wb')
pickle.dump(un_freq,fw)
pickle.dump(bi_freq,fw)