from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from graph_tool.all import *
import matplotlib.pylab as plt

import numpy as np
import networkx as nx
import scipy.linalg as lalg
import scipy.sparse.linalg as linalg
import scipy
import scipy.stats as stats
import pandas as pd
import vector as vc
import traceback
from scipy.sparse import lil_matrix, csr_matrix

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


def transition_matrix(M):
    n, n = M.shape
    P = np.copy(M)
    P /= P.sum(axis=1)
    return P


def leading_eigenvector(M, symmetric=False, overwrite_a=False, tol=0, max_inc_tol_fac=0, k=1):
    print 'largest eigenvec',
    k = min(k, M.shape[0] - 2)
    if symmetric:
        pass
    if scipy.sparse.issparse(M):
        while True:
            try:
                print 'sparse',
                if symmetric:
                    print 'symmetric'
                    l, v = linalg.eigsh(M, k=k, which="LA")
                else:
                    print 'asymmetric'
                    l, v = linalg.eigs(M, k=k, which="LR")  # maxiter=100) #np.iinfo(np.int32).max)
                l1 = l.real
                u = v[:, 0].real
                return l1, u / u.sum()
            except Exception as e:
                print traceback.format_exc()
                tol += np.finfo(float).eps
                if tol > np.finfo(float).eps * max_inc_tol_fac:
                    print 'no eigvec found. retry dense mode...'
                    return leading_eigenvector(M.todense(), overwrite_a=True)
                else:
                    print 'no eigvec found. retry with increased tol:', tol
                    return leading_eigenvector(M, symmetric=symmetric, tol=tol, max_inc_tol_fac=max_inc_tol_fac,
                                               overwrite_a=overwrite_a)
    else:
        print 'dense',
        if symmetric:
            print 'symmetric'
            l, v = lalg.eigh(M, overwrite_a=overwrite_a)
        else:
            print 'asymmetric'
            l, v = lalg.eig(M, overwrite_a=overwrite_a)
        l1index = largest_eigenvalue_index(l)
        u = np.array(vc.real_part(v[:, l1index]))
        return l[l1index].real, u


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


def katz_matrix(A, alpha, norm=None):
    m, n = A.shape
    if norm is None:
        katz = lil_matrix(np.eye(n)) - alpha * A
    elif len(norm.shape) == 1:
        katz = lil_matrix(np.diag(norm)) - alpha * A
    elif len(norm.shape) == 2:
        if not scipy.sparse.isspace(norm):
            norm = lil_matrix(norm)
        katz = norm - alpha * A
    else:
        print 'katz norm unknown shape'.center(120, '!')
        exit()
    return katz


def largest_eigenvalue_index(l):
    lreal = [a.real for a in l]
    l1index = np.argmax(lreal)
    return l1index


def calc_katz_iterative(A, alpha, max_iter=2000, filename='katz_range', out_dir='output/', plot=True, verbose=0):
    if verbose > 0:
        print 'calc katz iterative'
        print 'alpha:', alpha
    sigma = np.identity(A.shape[0])
    A_max, alphas = list(), list()
    orig_A = A.copy()
    orig_alpha = alpha
    for i in range(1, max_iter):
        if verbose > 1:
            print 'iter:', i
        if i > 1:
            A *= orig_A
            alpha *= orig_alpha
        M = np.multiply(A, alpha)
        sigma += M
        A_max.append(M.max())
        alphas.append(alpha)
        if np.allclose(A_max[-1], 0):
            if verbose > 0:
                print '\tbreak after length:', i
            break
    if plot:
        df = pd.DataFrame(columns=['max matrix value'], data=A_max)
        df['alpha'] = alphas
        df.plot(secondary_y=['alpha'], alpha=0.75, lw=2, logx=True, logy=True)
        plt.xlabel('path length')
        plt.ylabel('value')
        plt.savefig(out_dir + filename + '.png', bbox='tight')
        plt.close('all')
    return sigma


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


def rw_entropy_rate(M):
    n, n = M.shape
    P = transition_matrix(M)
    l, v = leading_eigenvector(P.T)
    entropy = 0.
    for i in range(n):
        row_sum = sum(M[i, :])
        if row_sum > 0:
            entropy += v[i] * stats.entropy(P[i, :], base=2)
    return entropy


def degree_product_matrix(A):
    d = deg_vector(A)
    return np.outer(d, d)
