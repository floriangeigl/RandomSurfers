__author__ = 'dhelic'

import numpy as np
import networkx as nx
import scipy.linalg as lalg
import scipy.sparse.linalg as linalg

import vector as vc


def adj_matrix(G, nodelist):
    A = nx.adjacency_matrix(G, nodelist)
    A = A.todense()
    A = np.asarray(A)
    return np.transpose(A)

def deg_vector(A):
    d = np.sum(A, axis=0)
    d[d == 0] = 1
    return d

def deg_matrix(A):
    d = deg_vector(A)
    D = np.diag(d)
    return D

def laplacian_matrix(A, D):
    return D - A

def deg_matrix_inv(A):
    d = deg_vector(A)
    di = [1/float(a) for a in d]
    Di = np.diag(di)
    return Di

def rwalk_matrix(A, D):
    Di = deg_matrix_inv(A)
    P = np.dot(A, Di)
    return P

def katz_alpha(A):
    lm = lmax(A)
    print "lmax%f"%lm
    alpha = (1/lm) * 0.15
    print "alpha%f"%alpha
    return alpha

def katz_matrix(A, alpha):
    m, n = A.shape
    katz = np.eye(n) - alpha * A
    return katz

def lmax(M):
    l,v = matrix_spectrum(M)
    lmax = l[0].real
    return lmax

def matrix_spectrum(M, sparse=True, k=1):
    if sparse:
        l, v = linalg.eigs(M, k=k, which="LR")
    else:
        l, v = lalg.eig(M)
    return l, v

def row_vector(M, index):
    x = M[index, :]
    return x

def column_vector(M, index):
    x = M[:, index]
    return x

def number_of_links(A, undirected=True):
    m = sum(sum(A))
    if undirected:
        return m / 2
    else:
        return m

def cosine_sim_matrix(A, undirected=True):
    sigma = np.zeros(A.shape)
    m, n = A.shape
    for i in xrange(n):
        x = nlinalg.row_vector(A, i)
        for j in xrange(n):
            y = nlinalg.row_vector(A, j)
            sim = vc.cosine(x, y)
            sigma[i][j] = sim
    return sigma

def katz_sim_matrix(A, alpha):
    katz = katz_matrix(A, alpha)
    sigma = lalg.inv(katz)
    return sigma

def modularity_matrix(A):
    m = number_of_links(A)
    Dp = degree_product_matrix(A)
    Dp *= 0.5 / m
    B = A - Dp
    B_max = deg_matrix(A) - Dp
    return B, B_max

def degree_product_matrix(A):
    d = deg_vector(A)
    return np.outer(d, d)
