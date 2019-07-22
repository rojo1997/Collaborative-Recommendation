import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

def readfile (name):
    """ Return a 3 columns matrix with links data
    - name: path to read the data
    The data is in csv format delimited by \t
    """
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

def multiGD(B):
    """ Return a matrix with the predictions of the ratings of the
    movies that users have not seen yet
    - B: links matrix
    - lr: learn rate
    - iter: max iterations
    - err: min error accepted
    It's a simple loop over the movies that calls the descending
    gradient and join all the data returned by DG.
    """
    # Compute Similarity matrix
    similarity = 1 - pairwise_distances(B, metric = "cosine")

    S = np.zeros(shape=B.shape, dtype=float)


    print(B.shape)

    score = 0.0

    # Arreglar score

    for i in range(B.shape[1]):
        X = np.append(np.ones([similarity.shape[1],1]),similarity,axis=1)
        Y = np.array(B[:,i])
        """X = X[Y!=0]
        Y = Y[Y!=0]"""
        clf = SGDClassifier(max_iter=5000, tol=1e-4)
        #clf = LinearSVC(random_state=0, tol=1e-5)
        print(i)
        if (len(Y[Y!=0]) == 0):
            S.T[i] = 2.5
        elif (np.std(Y[Y!=0]) == 0.0):
            S.T[i] = Y[Y!=0][0]
            score += len(Y[Y!=0])
        else:
            clf.fit(X[Y!=0],Y[Y!=0])
            S.T[i] = clf.predict(X)
            score += clf.score(X[Y!=0],Y[Y!=0]) * len(Y[Y!=0])
        
        S.T[i][Y!=0] = Y[Y!=0]
        

    print(score / len(B[B!=0]))

    return (S)

def error(S,T):
    """ Return error per diff rating movie that thay are new in test
    - S: matrix predictions
    - T: matrix test
    """
    return(np.sum(np.abs(S-T) * (T != 0))/np.count_nonzero(T))

def main():
    # READ DATA
    test = readfile (os.getcwd() + '/ml-100k/u2.test')
    base = readfile (os.getcwd() + '/ml-100k/u2.base')

    print("Test set size: ", len(test))
    print("Train set size: ", len(base))

    # COMPUTE MATRIX LINKS
    B = np.zeros(shape=(943,1682))
    for row in base:
        B[row[0]][row[1]] = row[2]
    T = np.zeros(shape=(943,1682))
    for row in test:
        T[row[0]][row[1]] = row[2]

    # KNN ALGORITHM + CROSS VALIDATION
    """neighbors = 15
    S = knnAlgorithm(B, neighbors)
    print("ERROR KNN TEST: ", error(S,T))

    k = 10
    S, err = CV(k, base, knnAlgorithm,neighbors)

    print("ERROR KNN CV: ", err)

    # KNN PREDICT
    user = 45
    n_movies = 3
    indexs = np.argpartition(S[user], -n_movies)[-n_movies:]
    print(indexs)

    # MULTI DESCENDING GRADIENT + CROSS VALIDATION"""
    S = multiGD(B.T)
    print("ERROR MDGD TEST: ", error(S.T,T))

    """k = 10
    S, err = CV(k, base, multiGD, lr, iter, err)

    print("ERROR DG CV: ", err)"""


if __name__ == '__main__':
    main()