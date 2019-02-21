import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import nltk

def edit_distance(x,y):
    n = len(x)
    m = len(y)
    dmatrix = np.zeros((n+1,m+1))
    for i in range(0,n+1):
        dmatrix[i,0] = i
    for j in range(1,m+1):
        dmatrix[0,j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            k = 0 if x[i-1]==y[j-1] else 2
            dmatrix[i,j] = min([dmatrix[i-1,j]+1,dmatrix[i,j-1]+1,
                dmatrix[i-1,j-1]+ k])
    return dmatrix[n,m]

with open('vocab.txt') as vocab_file:
    lines = vocab_file.readlines()
vocab = [line.strip() for line in lines]

testdata = pd.read_table('testdata.txt', header=None)
n = testdata.shape[0]

for i in range(0,n):
    misspell_count = 0
    for word in nltk.word_tokenize(testdata[2][i]):
        if misspell_count == testdata[1][i]:
           break
        if word not in vocab:
            print(word,end=' ')
            misspell_count += 1
    print()