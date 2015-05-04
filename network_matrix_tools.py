import numpy as np
import scipy
import linalg as la
import scipy.linalg as lalg
import scipy.stats as stats
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.preprocessing import normalize


def calc_common_neigh(adjacency_matrix):
    com_neigh = adjacency_matrix.dot(adjacency_matrix).todense()
    np.fill_diagonal(com_neigh, 0)
    return com_neigh


def calc_cosine(adjacency_matrix, weight_direct_link=False):
    if weight_direct_link:
        adjacency_matrix = adjacency_matrix.copy() + np.eye(adjacency_matrix.shape[0])
    com_neigh = adjacency_matrix.dot(adjacency_matrix)
    deg = adjacency_matrix.sum(axis=1).astype('float')
    deg_norm = np.sqrt(deg * deg.T)
    com_neigh /= deg_norm
    assert np.all(np.isfinite(com_neigh))
    return com_neigh


def katz_sim_network(adjacency_matrix, largest_eigenvalue, gamma=0.99):
    alpha_max = 1. / largest_eigenvalue
    alpha = gamma * alpha_max
    try:
        katz = la.katz_matrix(adjacency_matrix, alpha)
        sigma = lalg.inv(katz)
        return sigma
    except:
        return la.calc_katz_iterative(adjacency_matrix, alpha, plot=False)


def stationary_dist(transition_matrix):
    normed_transition_matrix = normalize(transition_matrix, norm='l1', axis=0, copy=True)
    assert np.all(normed_transition_matrix.data > 0)
    assert np.all(np.isfinite(normed_transition_matrix.data))
    eigval, stat_dist = la.leading_eigenvector(normed_transition_matrix)
    # print str(eigval).center(80, '*')
    assert np.all(np.isfinite(stat_dist))
    if not np.allclose(stat_dist, normed_transition_matrix * stat_dist):
        eigvals, _ = la.leading_eigenvector(normed_transition_matrix, k=10)
        print '=' * 80
        print eigvals
        print '=' * 80
        exit()
    # assert np.all(stat_dist > -0.1)
    while not np.isclose(stat_dist.sum(), 1.):
        stat_dist /= stat_dist.sum()
    # assert np.all(stat_dist > -0.0001)
    assert np.isclose(stat_dist.sum(), 1.)
    return stat_dist


def normalize_mat(matrix, replace_nans_with=0):
    if np.count_nonzero(matrix) == 0:
        print '\tnormalize all zero matrix -> set to all 1 before normalization'
        matrix = np.ones(matrix.shape, dtype='float')
        # np.fill_diagonal(M, 0)
    matrix /= matrix.sum(axis=1)  # copies sparse matrix
    if replace_nans_with is not None:
        matrix_sum = matrix.sum()
        if np.isnan(matrix_sum) or np.isinf(matrix_sum):
            print 'warn replacing nans with zero'
            matrix[np.invert(np.isfinite(matrix))] = replace_nans_with
    assert np.all(np.isfinite(matrix))
    return matrix


def calc_entropy_and_stat_dist(adjacency_matrix, bias=None):
    if bias is not None:
        if np.count_nonzero(bias) == 0:
            print '\tall zero matrix as weights -> use ones-matrix'
            bias = lil_matrix(np.ones(bias.shape))
        if len(bias.shape) == 1:
            bias_m = lil_matrix(adjacency_matrix.shape)
            bias_m.setdiag(bias)
            bias = bias_m
            weighted_trans = bias.tocsr() * adjacency_matrix
        elif len(bias.shape) == 2 and bias.shape[0] > 0 and bias.shape[1] > 0:
            weighted_trans = adjacency_matrix.multiply(lil_matrix(bias))
        else:
            print '\tunknown bias shape'
    else:
        weighted_trans = adjacency_matrix.copy()
    # weighted_trans = normalize_mat(weighted_trans)
    stat_dist = stationary_dist(weighted_trans)
    return entropy_rate(weighted_trans, stat_dist=stat_dist), stat_dist


def entropy_rate(transition_matrix, stat_dist=None, base=2):
    if stat_dist is None:
        stat_dist = stationary_dist(transition_matrix)
    if scipy.sparse.issparse(transition_matrix):
        transition_matrix = transition_matrix.todense()
    entropies = stats.entropy(transition_matrix, base=base) * stat_dist
    rate = np.sum(entropies[np.isfinite(entropies)])
    assert np.isfinite(rate)
    return rate