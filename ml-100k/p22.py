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

def knnAlgorithm (B,neighbors):
    """ Return a matrix with the predictions of the ratings of the
    movies that users have not seen yet
    - B: links matrix
    - neighbors: number of similar users that you consider in a
    prediction
    KNN algorithm is a non-parametic method used for classification
    regresion.
    """
    # Compute Similarity matrix using links matrix
    similarity = 1 - pairwise_distances(B, metric = "cosine")
    # Store matrix indexs
    max_indexs = np.zeros(shape=(len(B),neighbors), dtype=int)
    # k-largest indexs to optimize calculations
    for i in range(0,len(B)):
        max_indexs[i] = np.argpartition(similarity[i], -neighbors)[-neighbors:]
    # Return matrix object
    S = np.zeros(shape=(len(B),1682), dtype=float)
    # For each user
    for i in range(0,B.shape[0]):
        # For each movie
        for j in range(0,B.shape[1]):
            # If you have seen it
            if (B[i][j] == 0):
                # We take the top k similary from user i
                values = B.T[j][max_indexs[i]]
                # We remove non rate movies
                values = values[values != 0]
                # If the set is empty we predict the middle value
                if len(values) == 0:
                    S[i][j] = 2.5
                # Else we do the mean
                else:
                    S[i][j] = np.mean(values)

    return(S)

def GDAlgorithm(X, Y, w, lr, iter, err, per):
    """ Return a coefficient, labels vector and error from one movie
    - X: similarity matrix between users
    - Y: ratings of a movie
    - lr: learn rate
    - iter: max iterations
    - err: min error accepted
    - per: percentage stochastic
    Gradient descent is a first-order iterative optimization algorithm
    for finding the minimun of a function.
    """
    # Loop iterator
    i = 0
    # Loop logit condition
    test = True
    # Var if the algorithm were stochastic
    w_best = w
    y_best = np.zeros(shape=(X.shape[1] - 1,1))
    er_best = 4
    er = 4
    m = Y.shape[1]
    while (i < iter and test):
        # Compute gradient
        Y_new = np.dot(X,w)
        """max = np.max(Y_new)
        min = np.min(Y_new)
        Y_new = 1 + 4 * (Y_new - min) / (max - min)"""
        w = w - (1/m) * lr * X.T.dot((Y_new - Y.T))

        # Compute error
        aux = np.abs(Y_new.T - Y)
        er = np.sum(aux[Y != 0]) / np.count_nonzero(Y)
        # Text improve
        if (er < er_best):
            er_best = er
            w_best = w
            y_best = Y_new
            #print(er)
        if (er < err):
            test = False
        # Iteration in loop
        i += 1
    return(er_best, y_best.T)

def multiGD(B, lr = 0.1, iter = 100, err = 0.5, per = 0.2):
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
    # Coefficients matrix
    W = np.zeros(shape=(B.shape[1],B.shape[0]+1), dtype=float)
    # Predictions matrix
    S = np.zeros(shape=B.shape, dtype=float)
    # Sum error for each movie
    some = 0.0
    # For each movie
    for i in range(B.shape[1]):
        # We decompose the data to form the function
        X = np.append(np.ones([similarity.shape[1],1]),similarity,axis=1)
        Y = np.matrix(B[:,i])
        # We call gradient descending
        w = np.ones(shape = (B.shape[0] + 1,1))
        er, S.T[i] = GDAlgorithm(X = X, Y = Y, w = w, lr = lr, iter = iter, err = err, per = per)
        some += er
        if (i % 50) == 0:
            print(str(i) + ": " + str(some/(i+1)))
    er = some / B.shape[1]
    return (S)

def CV(k, base, func, *args):
    """ Return a matrix with the predictions of the ratings of the
    movies that users have not seen yet and mean error
    - k: number of folds
    - B: B: links table
    - func: predition method
    - args: arguments of the method
    """
    # Number of data per fold
    size = int(len(base) / k)
    # Generic error
    som = 0.0
    # Recall error
    rec = 0.0
    # For each fold
    for i in range(0,k):
        # Split set
        test_val = base[i * size:i * size + size]
        train_val = base[0:i * size] + base[i * size + size:len(base)]
        # Create links matrix train and test
        # Links matrix
        B_train = np.zeros(shape=(943,1682))
        B_test = np.zeros(shape=(943,1682))
        for row in train_val:
            B_train[row[0]][row[1]] = row[2]
        for row in test_val:
            B_test[row[0]][row[1]] = row[2]
        # Recall part (delete 4 links for each users)
        B_train_recall, D = recall(B_train,4)
        # Algorithm call
        S = func(B_train_recall,*args)
        # Recall part (predict links)
        for j in range(0,B_train_recall.shape[0]):
            indexs = np.argpartition(S[j], -20)[-20:]
            inter = list(set(indexs) & set(D[j]))
            rec += (len(inter) / 4)
        print("Recall fold " + str(i) + ": " + str(rec / B_train_recall.shape[0]))
        print("Generic error fold " + str(i) + ": " + str(error(S,B_test)))
        # Accumulate error
        som = som + error(S,B_test)
    # Mean error
    err = som / k
    return (S, err)

def error(S,T):
    """ Return error per diff rating movie that thay are new in test
    - S: matrix predictions
    - T: matrix test
    """
    return(np.sum(np.abs(S-T) * (T != 0))/np.count_nonzero(T))

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
        select = indexs[B[i] >= 4]
        # Randon selection
        rd.shuffle(select)
        select = select[:k]
        if len(select) == k:
            D[i] = select
            # Delete links
            B[i,select] = 0
    return (B,D)

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
    neighbors = 15
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

    # MULTI DESCENDING GRADIENT + CROSS VALIDATION
    lr = 0.1
    iter = 1000
    err = 0.5
    per = 0.2
    S = multiGD(B, lr = lr, iter = iter, err = err, per = per)
    print("ERROR DG TEST: ", error(S,T))

    k = 10
    S, err = CV(k, base, multiGD, lr, iter, err, per)

    print("ERROR DG CV: ", err)

if __name__ == '__main__':
    main()