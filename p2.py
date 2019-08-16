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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

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

def genClf(B, n, clf, row=True, T = 0):
    S, P, _ = multiClf(B, clf, row)
    print("ERROR MDGD TEST: ", mae(S,T))
    for i in range(n):
        k = np.percentile(P[S != 0][P[S != 0] != 0], 99.99)
        S[P < k] = 0
        S[B != 0] = 0
        B += S
        print("DESPUES: ", len(B[B > 0]))
        S, P, _ = multiClf(B, clf, row)
        print("ERROR MDGD TEST: ", mae(S,T))
        print(i)
    return (S, P)

# Añadir por filas o columnas
# https://www.youtube.com/watch?v=h9gpufJFF-0
def multiClf(B, clf, row = True, n = 0):
    """ Return a matrix with the predictions of the ratings of the
    movies that users have not seen yet
    - B: links matrix
    - clf: algorithm choice
    - row: apply the algorithm
    - iter: max iterations
    - tol: tolerance for stopping criteria
    It's a simple loop over the movies that calls the descending
    gradient and join all the data returned by DG.
    """
    if not row: B = B.T

    # Compute Similarity matrix
    similarity = 1 - pairwise_distances(B, metric = "cosine")
    #similarity = 1 - pairwise_distances(B, metric = "euclidean")
    #similarity = 1 - pairwise_distances(B, metric = "cityblock")

    S = np.zeros(shape=B.shape, dtype=float)
    P = np.zeros(shape=B.shape, dtype=float)

    score = 0.0

    for i in range(B.shape[1]):
        if (i % 200 == 0): print(i)
        X = np.append(np.ones([similarity.shape[1],1]),similarity,axis=1)
        Y = np.array(B[:,i])
        if (len(Y[Y!=0]) == 0 or ((n != 0) and (n > len(Y[Y!=0])))):
            S.T[i] = 2.5
            P.T[i] = 0
        elif (np.std(Y[Y!=0]) == 0.0):
            S.T[i] = Y[Y!=0][0]
            score += len(Y[Y!=0])
            P.T[i] = 0
        else:
            clf.fit(X[Y!=0],Y[Y!=0])
            S.T[i] = clf.predict(X)
            s = clf.score(X[Y!=0],Y[Y!=0]) * len(Y[Y!=0])
            P.T[i] = s
            score += s
        
        S.T[i][Y!=0] = Y[Y!=0]
        
    S[B != 0] = 0
    score = score / len(B[B!=0])
    if not row:
        B = B.T
        S = S.T
        P = P.T

    return (S,P, score)

def mae(S,T):
    """ Return mean absolute error
    - S: matrix predictions
    - T: matrix test
    """
    return(np.sum(np.abs(S-T) * (T != 0))/np.count_nonzero(T))

def rmse(S,T):
    """ Return mean square error
    - S: matrix predictions
    - T: matrix test
    """
    return(np.sqrt(np.sum(np.abs(S-T) * np.abs(S-T) * (T != 0))/np.count_nonzero(T)))

def perc(S,T):
    """ Return per rating movie that thay are new in test
    - S: matrix predictions
    - T: matrix test
    """
    return(np.sum((S == T) * (T != 0))/np.count_nonzero(T))

def rc(base,k,s):
    """ Return matrix links with some random links deleted and a matrix
    with the indexs of this links
    B: matrix links
    k: number of links to delete
    """
    BN = np.zeros(shape=(943,1682))
    baseN = base.copy()
    np.random.shuffle(baseN)
    baseN = np.delete(baseN,np.arange(0,k,1), axis=0)
    for row in baseN:
        BN[row[0]][row[1]] = row[2]
    return(BN)

def CV(base, test, clf, k, n = 0):
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(base)
    metrix = np.zeros(shape=(k,7))
    i = 0
    B = np.zeros(shape=(943,1682))
    T = np.zeros(shape=(943,1682))
    for base_index, test_index in kf.split(base):
        print(i)
        base_cv = base[base_index]
        test_cv = base[test_index]
        B.fill(0); T.fill(0)
        for row in base_cv:
            B[row[0]][row[1]] = row[2]
        for row in test_cv:
            T[row[0]][row[1]] = row[2]
        s = 3
        BN = rc(base_cv,20,s)
        print(len(BN[BN!=B]))
        S,_, score = multiClf(B=BN, clf=clf, row=False, n = n)
        recall_aprox = len(S[BN != B][S[BN != B] == B[BN != B]]) / 20
        precision = 0.0
        recall = 0.0
        for j in range(1,6):
            precision += len(S[S == T][S[S == T] == j]) / len(S[S == j])
            recall += len(S[S == T][S[S == T] == j]) / len(T[T == j])
        precision /= 5; recall /= 5
        metrix[i] = np.array([score, mae(S,T), rmse(S,T), perc(S,T), precision, recall, recall_aprox])
        i += 1
    return (metrix)

def main():
    # READ DATA
    test = np.array(readfile (os.getcwd() + '/ml-100k/u2.test'))
    base = np.array(readfile (os.getcwd() + '/ml-100k/u2.base'))

    print("Test set size: ", len(test))
    print("Train set size: ", len(base))

    # COMPUTE MATRIX LINKS
    B = np.zeros(shape=(943,1682))
    for row in base:
        B[row[0]][row[1]] = row[2]
    T = np.zeros(shape=(943,1682))
    for row in test:
        T[row[0]][row[1]] = row[2]

    # CROSS VALIDATION
    k = 10
    neighbors = 15

    clf = KNeighborsClassifier(n_neighbors = neighbors)
    metrix = CV(base = base, test = test, clf = clf, k = k, n = neighbors)
    print(metrix)
    print(np.mean(metrix, axis=0))

    clf = SGDClassifier(max_iter=3000, tol=1e-4)
    metrix = CV(base, test, clf, k)
    print(metrix)
    print(np.mean(metrix, axis=0))

    clf = RandomForestClassifier(n_estimators=100, n_jobs=1)
    metrix = CV(base, test, clf, k)
    print(metrix)
    print(np.mean(metrix, axis=0))

if __name__ == '__main__':
    main()