import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.linear_model import LinearRegression

def readfile (name):
    array = []
    csv.register_dialect('myDialect',
        delimiter = '\t',
        skipinitialspace=True
    )
    with open(name) as csvFile:
        reader = csv.reader(csvFile, dialect = 'myDialect')
        for row in reader:
            array.append([int(row[0])-1,int(row[1])-1,int(row[2])])
    csvFile.close()
    return (array)

def knnAlgorithm (B,neighbors):
    # Compute Similarity matrix between
    similarity = 1 - pairwise_distances(B, metric = "cosine")
    # Store matrix indexs
    max_indexs = np.zeros(shape=(len(B),neighbors), dtype=int)
    # k-largest indexs
    for i in range(0,len(B)):
        max_indexs[i] = np.argpartition(similarity[i], -neighbors)[-neighbors:]
    S = np.zeros(shape=(len(B),1682), dtype=float)
    for i in range(0,len(B)):
        for j in range(0,1682):
            if (B[i][j] == 0):
                values = B.T[j][max_indexs[i]]
                values = values[values != 0]
                if len(values) == 0:
                    S[i][j] = 2.5
                else:
                    S[i][j] = np.mean(values)

    return(S)

def error(S,T):
    return(np.sum(np.abs(S-T) * (T != 0))/np.count_nonzero(T))

def cross_validation_knn(k, B, neighbors):
    size = int(B.shape[0] / k)
    print("size split: ",size)
    som = 0.0
    rec = 0.0
    for i in range(0,k):
        # Split set
        test_val = B[i * size:i * size + size]
        train_val = np.vstack((B[0:i * size], B[i * size + size:B.shape[0]]))
        # Recall part (delete links)
        train_recall, D = recall (train_val,4)
        # Algorithm call
        S = knnAlgorithm(train_recall,neighbors)
        # Recall part (predict links)
        for j in range(0,train_val.shape[0]):
            indexs = np.argpartition(S[j], -4)[-4:]
            inter = list(set(indexs) & set(D[j]))
            rec += (len(inter) / 4)
        print(rec / train_val.shape[0])
        print(S.shape)
        print(test_val.shape)
        som = som + error(S,test_val)
    return (som/k)

def cross_validation_DG(k, B, lr, iter, err):
    size = int(B.shape[0] / k)
    print("size split: ",size)
    som = 0
    recall_som = 0
    for i in range(0,k):
        test_val = B[i * size:i * size + size]
        train_val = np.vstack((B[0:i * size], B[i * size + size:B.shape[0]]))
        train_recall, D = recall (train_val,4)
        S,W,err = multiDG(train_val, lr = lr, iter = iter, err = err)
        for j in range(0,B.shape[0]):
            indexs = np.argpartition(S[j], -4)[-4:]
            rec = len(indexs.intersection(D[j])) / 4
        print(rec / B.shape[0])
        som = som + error(S,test_val)
    return (som/k)

def sigma(x):
    return 1 / (1 + np.exp(-x))

def sumGD(X,Y,w,j):
    acc = 0.0
    for n in range(0,X.shape[0]):
        acc = acc + X[n,j] * (np.sign(np.sum(w * X[n])) - Y[:,n])
    return (acc)

def evalu(X,Y,w):
    Y_new = np.matmul(w,X.T)
    max = np.max(Y_new)
    min = np.min(Y_new)
    Y_new = 1 + 4 * (Y_new - min) / (max - min)
    err = error(Y_new.T[0:Y_new.shape[1] - 1], Y)
    return(err)

def GD(X, Y, lr, iter, err):
    i = 0
    test = True
    # Matrix (n * (m + 1))
    w = np.ones(shape=(1,X.shape[1]), dtype=float)
    w_best = w
    y_best = w
    er_best = 4
    while (i < iter and test):
        # (n * (m + 1)) = (n * (m + 1)) - k * (n * (m + 1))
        for j in range(0,X.shape[0]):
            w[:,j] = w[:,j] - lr * sumGD(X,Y,w,j)
        i += 1
        # k1 < k2
        Y_new = np.matmul(w,X.T)
        max = np.max(Y_new)
        min = np.min(Y_new)
        Y_new = 1 + 4 * (Y_new - min) / (max - min)
        er = np.sum(np.multiply(np.abs(Y_new - Y), (Y != 0))) / np.count_nonzero(Y)
        if (er < er_best):
            er_best = er
            w_best = w
            y_best = Y_new
        if (er < err or er == err):
            test = False
    return(w_best, er_best, y_best)

def multiDG(B, lr, iter, err):
    similarity = 1 - pairwise_distances(B, metric = "cosine")
    W = np.zeros(shape=(B.shape[1],B.shape[0]+1), dtype=float)
    S = np.zeros(shape=B.shape, dtype=float)
    some = 0.0
    for i in range(B.shape[1]):
        X = np.append(np.ones([similarity.shape[1],1]),similarity,axis=1)
        Y = np.matrix(B[:,i])
        W[i], err, S.T[i] = GD(X = X, Y = Y, lr = lr, iter = iter, err = err)
        print(err)
        some = some + err
    err = some / B.shape[1]
    return (S,W,err)

def recall(B,k):
    D = np.zeros(shape=(B.shape[0],k))
    indexs = np.arange(0,B.shape[1],1)
    for i in range(B.shape[0]):
        select = indexs[B[i] >= 3]
        rd.shuffle(select)
        select = select[:k]
        D[i] = select
        B[i,select] = 0
    return (B,D)

def main():
    # READ DATA
    test = readfile (os.getcwd() + '/ml-100k/u2.test')
    base = readfile (os.getcwd() + '/ml-100k/u2.base')

    print(len(test))
    print(len(base))

    # COMPUTE MATRIX LINKS

    B = np.zeros(shape=(943,1682))
    for row in base:
        B[row[0]][row[1]] = row[2]
    T = np.zeros(shape=(943,1682))
    for row in test:
        T[row[0]][row[1]] = row[2]

    # KNN ALGORITHM + CROSS VALIDATION
    neighbors = 15

    S = knnAlgorithm(B, neighbors)

    print("ERROR KNN IN: ", error(S,B))
    print("ERROR KNN OUT: ", error(S,T))
    
    k = 10

    err = cross_validation_knn(k, B, neighbors)

    # KNN PREDICT
    user = 45
    n_movies = 3
    indexs = np.argpartition(S[user], -n_movies)[-n_movies:]
    print(indexs)
    

    # MULTI DESCENDING GRADIENT + CROSS VALIDATION
    """lr = 0.1
    iter = 100
    err = 0.5

    S, w, err = multiDG(B, lr = lr, iter = iter, err = err)

    k = 10

    err = cross_validation(k, B, lr = lr, iter = iter, err = err)

    print("ERROR DG: ", error(S,T))"""

    #np.savetxt('B.txt',B,fmt = '%.2f')
    #np.savetxt('S.txt',S,fmt = '%.2f')


    # Nota: 
    """model = LinearRegression()
    model.fit(X,Y)"""

    


if __name__ == '__main__':
    main()