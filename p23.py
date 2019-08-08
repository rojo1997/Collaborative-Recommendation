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
    S, P = multiClf(B, clf, row)
    print("ERROR MDGD TEST: ", error(S,T))
    for i in range(n):
        k = np.percentile(P[S != 0][P[S != 0] != 0], 99.99)
        S[P < k] = 0
        S[B != 0] = 0
        B += S
        print("DESPUES: ", len(B[B > 0]))
        S, P = multiClf(B, clf, row)
        print("ERROR MDGD TEST: ", error(S,T))
        print(i)
    return (S, P)

# Añadir por filas o columnas
# https://www.youtube.com/watch?v=h9gpufJFF-0
def multiClf(B, clf, row = True):
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
        X = np.append(np.ones([similarity.shape[1],1]),similarity,axis=1)
        Y = np.array(B[:,i])
        if (len(Y[Y!=0]) == 0):
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

def error(S,T):
    """ Return error per diff rating movie that thay are new in test
    - S: matrix predictions
    - T: matrix test
    """
    return(np.sum(np.abs(S-T) * (T != 0))/np.count_nonzero(T))

def perc(S,T):
    """ Return per rating movie that thay are new in test
    - S: matrix predictions
    - T: matrix test
    """
    return(np.sum((S == T) * (T != 0))/np.count_nonzero(T))

def recall(B,k):
    """ Return matrix links with some random links deleted and a matrix
    with the indexs of this links
    B: matrix links
    k: number of links to delete
    """
    # Matrix of indexs
    D = np.zeros(shape=(B.shape[0],k))
    # Indexs selections
    indexs = np.arange(0,B.shape[1],1)
    for i in range(B.shape[0]):
        # We consider only good ratings to be deleted
        select = indexs[B[i] >= 3]
        # Randon selection
        rd.shuffle(select)
        select = select[:k]
        if len(select) == k:
            D[i] = select
            # Delete links
            B[i,select] = 0
    return (B,D)

def CV(base, test, clf, k):
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(base)
    metrix = np.zeros(shape=(k,3))
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
        S,_, score = multiClf(B=B, clf=clf, row=False)
        metrix[i] = np.array([score, error(S,T), perc(S,T)])
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
    #clf = GaussianNB()
    #clf = SVC(gamma='auto')
    #clf = RandomForestClassifier(n_estimators=100, n_jobs=1)
    #clf = AdaBoostClassifier(n_estimators=100)
    #clf = DecisionTreeClassifier(random_state=0)
    #clf = QuadraticDiscriminantAnalysis()
    """S,P, score = multiClf(B=B, clf=clf, row=False)
    print("SCORE: ", score)
    print("ERROR MDGD TEST: ", error(S,T))
    print("PERC MDGD TEST: ", perc(S,T))"""

    # CROSS VALIDATION
    k = 10
    clf = SGDClassifier(max_iter=3000, tol=1e-4)
    metrix = CV(base, test, clf, k)
    print(metrix)
    print("score train: ", np.average(metrix[:,0]))
    print("error abs: ", np.average(metrix[:,1]))
    print("perc: ", np.average(metrix[:,2]))

    clf = RandomForestClassifier(n_estimators=100, n_jobs=1)
    metrix = CV(base, test, clf, k)
    print(metrix)
    print("score train: ", np.average(metrix[:,0]))
    print("error abs: ", np.average(metrix[:,1]))
    print("perc: ", np.average(metrix[:,2]))

    clf = LinearSVC()
    metrix = CV(base, test, clf, k)
    print(metrix)
    print("score train: ", np.average(metrix[:,0]))
    print("error abs: ", np.average(metrix[:,1]))
    print("perc: ", np.average(metrix[:,2]))


    """kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(base)
    
    score_sum = 0.0
    error_sum = 0.0
    perc_sum = 0.0
    for base_index, test_index in kf.split(base):
        print("TRAIN:", len(base_index), "TEST:", len(test_index))
        base_cv = base[base_index]
        test_cv = base[test_index]
        B = np.zeros(shape=(943,1682))
        for row in base_cv:
            B[row[0]][row[1]] = row[2]
        T = np.zeros(shape=(943,1682))
        for row in test_cv:
            T[row[0]][row[1]] = row[2]
        S,P, score = multiClf(B=B, clf=clf, row=False)
        score_sum += score
        error_sum += error(S,T)
        perc_sum += perc(S,T)
    print("SCORE: ", score_sum/k)
    print("ERROR MDGD TEST: ", error_sum/k)
    print("PERC MDGD TEST: ", perc_sum/k)"""

    """k = 10
    S, err = CV(k, base, multiGD, lr, iter, err)

    print("ERROR DG CV: ", err)"""


if __name__ == '__main__':
    main()