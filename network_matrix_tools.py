import numpy as np
import scipy
import linalg as la
import scipy.linalg as lalg
import scipy.stats as stats
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, diags
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from sklearn.preprocessing import normalize
import traceback
from scipy.sparse.csgraph import connected_components
import utils
from tools import gt_tools
from graph_tool.all import *
import operator
import multiprocessing

def calc_common_neigh(adjacency_matrix):
    com_neigh = lil_matrix(adjacency_matrix.dot(adjacency_matrix))
    com_neigh.setdiag(0)
    return com_neigh

def calc_cosine(adjacency_matrix, weight_direct_link=False):
    if weight_direct_link:
        b = adjacency_matrix + sparse.eye(adjacency_matrix.shape[0])
    else:
        b = adjacency_matrix
    deg = adjacency_matrix.sum(axis=0)
    cos = lil_matrix(adjacency_matrix.dot(b))
    cos.setdiag(np.array(deg).flatten())
    deg = csr_matrix(deg)
    cos = cos.tocsr()
    deg_norm = np.sqrt(deg.T.dot(deg))
    cos = cos.multiply(csr_matrix(1. / deg_norm))
    cos.data[np.invert(np.isfinite(cos.data))] = 0
    cos.eliminate_zeros()
    # cos.setdiag(1.)
    assert np.all(np.isfinite(cos.data))
    return cos


def katz_sim_network(adjacency_matrix, largest_eigenvalue, gamma=0.99, norm=None, mask_adj=True):
    alpha_max = 1. / largest_eigenvalue
    alpha = gamma * alpha_max
    try:
        sigma = la.katz_matrix(adjacency_matrix, alpha, norm=norm)
        if scipy.sparse.issparse(sigma):
            sigma = sigma.todense()
        sigma = lalg.inv(sigma, overwrite_a=True)
        if norm is not None:
            if len(norm.shape) == 1:
                sigma *= scipy.sparse.spdiags(norm, 0, sigma.shape[0], sigma.shape[0])
            else:
                sigma *= norm
        print 'mask'
        if mask_adj:
            # memory efficient way
            sigma = sigma[adjacency_matrix.nonzero()]  # get relevant data and delete dense matrix
            sigma = np.array(sigma)  # convert to np.array and
            sigma = sigma.flatten()  # flatten it
            sigma = csr_matrix((sigma, adjacency_matrix.nonzero()),
                               shape=adjacency_matrix.shape)  # create csr matrix using indices and shape of sparse matrix
            sigma.eliminate_zeros()  # delete remaining zeros
        print 'done'
        return sigma
    except Exception as e:
        print traceback.format_exc()

        if norm is None:
            print 'use iterative katz'
            return la.calc_katz_iterative(adjacency_matrix, alpha, plot=False)
        else:
            print 'could not calc katz'.center(120, '!')
            raise Exception(e)


def pagerank_from_transmat(weighted_adj_matrix, print_prefix='', damping=0.85):
    net = gt_tools.net_from_adj(weighted_adj_matrix, parallel_edges=False)
    pi = np.array(pagerank(net, weight=net.ep['weights'], damping=damping).a)
    return pi / pi.sum()


def stationary_dist(transition_matrix, print_prefix='', atol=1e-10, rtol=0., scaling_factor=1e5):
    P = normalize(transition_matrix, norm='l1', axis=0, copy=True)
    P.data *= scaling_factor
    assert not np.any(P.data < 0)
    zeros_near_z = np.isclose(P.data, 0., rtol=0., atol=1e-10).sum()

    assert np.all(np.isfinite(P.data))
    components = connected_components(P, connection='strong', return_labels=False)
    if components > 1:
        print print_prefix, 'more than 1 component. eigenvector will not converge.'
        print print_prefix, 'try using pagerank instead!'
        raise scipy.sparse.linalg.ArpackNoConvergence
    print print_prefix, 'P values near zero: #', zeros_near_z
    try:
        eigval, pi = la.leading_eigenvector(P, print_prefix=print_prefix)  # , init_v=np.array(P.sum(axis=1)).flatten())
    except scipy.sparse.linalg.ArpackNoConvergence:
        print print_prefix, 'eigenvector did not converge!'
        print print_prefix, 'try using pagerank instead!'
        raise
        #raise scipy.sparse.linalg.ArpackNoConvergence

    assert np.all(np.isfinite(pi))
    normed_P = normalize(transition_matrix, norm='l1', axis=0, copy=True)
    if not np.allclose(pi, normed_P * pi, atol=atol, rtol=rtol) \
            or not np.isclose(eigval, scaling_factor, atol=atol*scaling_factor, rtol=rtol):
        # eigval, _ = la.leading_eigenvector(P, k=10, print_prefix=print_prefix)
        components = connected_components(P, connection='strong', return_labels=False)
        print print_prefix + 'pi = P * pi:', np.allclose(pi, normed_P * pi, atol=atol, rtol=rtol)
        print print_prefix + 'eigval == 1:', np.isclose(eigval, scaling_factor, atol=atol*scaling_factor, rtol=rtol)
        print print_prefix, '=' * 80
        if components > 1:
            print print_prefix, utils.color_string('# components: ' + str(components), utils.bcolors.RED)
        else:
            print '# components: ', components
        print print_prefix, "eigval: %.10f" % eigval.real[0]
        print print_prefix, '=' * 80
        exit()
    close_zero = np.isclose(pi, 0, atol=atol, rtol=rtol)
    neg_stat_dist = pi < 0
    pi[close_zero & neg_stat_dist] = 0.
    if np.any(pi < 0):
        # eigvals, _ = la.leading_eigenvector(P, k=10, print_prefix=print_prefix)
        components = connected_components(P, connection='strong', return_labels=False)
        eigval, _ = la.leading_eigenvector(P, k=10, print_prefix=print_prefix)
        print print_prefix + 'negative stat values:', list(map(lambda i: "%.10f" % i, pi[pi < 0]))[:10], '...'
        # print print_prefix + 'negative stat sum:', "%.10f" % pi[pi < 0].sum()
        # print print_prefix + 'negative stat max:', "%.10f" % pi[pi < 0].min()
        print print_prefix, '=' * 80
        print '# components: ', components
        print print_prefix, 'eigval:', eigval
        print print_prefix, '=' * 80
        raise scipy.sparse.linalg.ArpackNoConvergence
    while not np.isclose(pi.sum(), 1, atol=atol, rtol=rtol):
        print print_prefix, utils.color_string('re-normalize stat. dist.'.center(100, '!'), utils.bcolors.RED)
        pi /= pi.sum()
        close_zero = np.isclose(pi, 0, atol=atol, rtol=rtol)
        neg_stat_dist = pi < 0
        pi[close_zero & neg_stat_dist] = 0.
        assert not np.any(pi < 0)
    return pi


def calc_entropy_and_stat_dist(adjacency_matrix, bias=None, print_prefix='', eps=1e-10, orig_ma_mi_r=None, method='PR',
                               damping=0.85, smooth_bias=True):
    # supported methods:
    # PR: pagerank
    # EV: eigenvector
    bias_max_min_r = None
    weighted_trans = None
    if bias is not None:
        if np.count_nonzero(bias) == 0:
            print print_prefix + '\tall zero matrix as weights -> use ones-matrix'
            bias = lil_matrix(np.ones(bias.shape))
            bias_max_min_r = 1.
        if len(bias.shape) == 1:
            bias_m = diags(bias, 0)
            bias_max_min_r = bias.max() / bias.min()
            weighted_trans = bias_m.dot(adjacency_matrix)
        elif len(bias.shape) == 2 and bias.shape[0] > 0 and bias.shape[1] > 0:
            try:
                bias_max_min_r = (bias.max()) / (bias.min())
            except:
                bias_max_min_r = (bias.max()) / (bias.min())
            if bias.shape != adjacency_matrix.shape:
                print print_prefix + 'inconsistent shape:', bias.shape, adjacency_matrix.shape
            # mask bias (filter out relevant values)
            bias = csr_matrix(bias)
            bias.eliminate_zeros()
            bias = bias.multiply(adjacency_matrix.astype('bool'))
            # created weighted trans mat
            weighted_trans = adjacency_matrix.multiply(bias)
        else:
            print print_prefix + '\tunknown bias shape'
    else:
        weighted_trans = adjacency_matrix.copy()
    # weighted_trans.eliminate_zeros()
    # weighted_trans = normalize_mat(weighted_trans)
    stat_dist = None
    if method == 'EV':
        try:
            stat_dist = stationary_dist(weighted_trans, print_prefix=print_prefix)
            if orig_ma_mi_r is not None:
                print 'orig bias max/min:', orig_ma_mi_r
                print 'normalized max/min:', bias_max_min_r
        except scipy.sparse.linalg.ArpackNoConvergence as e:
            if smooth_bias:
                print print_prefix, 'no converge. add epsilon to bias', eps
                if bias is None:
                    raise e
                b_zeros = 0
                if bias is not None:
                    bias_o = np.float(10 ** int(np.ceil(np.log10(bias.shape[0]))))
                    add_eps = eps/bias_o
                    print print_prefix, 'absolute eps:', utils.color_string(str(add_eps), utils.bcolors.RED)
                    if len(bias.shape) == 1:
                        # print print_prefix, 'vector bias'
                        bias /= bias.sum()
                        b_zeros = np.isclose(bias, 0., rtol=0., atol=1e-15).sum() / len(bias)
                        bias += add_eps
                    else:
                        if scipy.sparse.issparse(bias):
                            # print print_prefix, 'sparse matrix bias'
                            bias = normalize(bias, 'l1', axis=0, copy=False)
                            b_zeros = np.isclose(bias.data, 0., rtol=0., atol=1e-15).sum() / len(bias.data)
                            bias.data += add_eps
                        else:
                            # print print_prefix, 'dense matrix bias'
                            bias /= bias.sum(axis=0)
                            b_zeros = np.isclose(np.array(bias).flatten(), 0., rtol=0., atol=1e-15).sum() / (
                            bias.shape[0] * bias.shape[1])
                            bias += add_eps
                print print_prefix, b_zeros * 100, '% of all values in bias near zero. '  # eps:', 1e-15
                eps *= 10
                return calc_entropy_and_stat_dist(adjacency_matrix, bias=bias, print_prefix=print_prefix, eps=eps,
                                                  orig_ma_mi_r=bias_max_min_r if orig_ma_mi_r is None else orig_ma_mi_r,
                                                  smooth_bias=True, method='EV')
            else:
                raise e
    elif method == 'PR':
        stat_dist = pagerank_from_transmat(weighted_trans, print_prefix=print_prefix, damping=damping)
    else:
        print print_prefix, 'unknown method:', method
    assert stat_dist is not None
    return entropy_rate(weighted_trans, stat_dist=stat_dist, print_prefix=print_prefix), stat_dist


def entropy_rate(weighted_adj_matrix, stat_dist=None, base=2, print_prefix=''):
    print print_prefix + 'calc entropy rate'
    if stat_dist is None:
        stat_dist = stationary_dist(weighted_adj_matrix)
    assert not np.any(stat_dist < 0)
    assert np.isclose(stat_dist.sum(), 1.)
    # assert np.all(weighted_adj_matrix.sum(axis=0) > 0)
    if scipy.sparse.issparse(weighted_adj_matrix):
        if not isinstance(weighted_adj_matrix, csc_matrix):
            weighted_adj_matrix = weighted_adj_matrix.tocsc()
        get_col = weighted_adj_matrix.getcol
        col_entropy = (get_col(i).data for i in xrange(weighted_adj_matrix.shape[0]))
        col_entropy = np.array(map(lambda x: stats.entropy(x, base=base), col_entropy)).flatten()
    else:
        col_entropy = np.array(stats.entropy(weighted_adj_matrix, base=base)).flatten()
    stat_dist = np.array(stat_dist).flatten()
    assert stat_dist.shape == col_entropy.shape
    col_entropy *= stat_dist
    finite_elements = np.isfinite(col_entropy)
    if not all(finite_elements):
        print print_prefix + 'WARN: entropy rate contains not finite elements. (inf, nan)'
    rate = np.sum(col_entropy[finite_elements])
    if not np.isfinite(rate):
        print print_prefix + 'entropy rate not finite'
        exit()
    return rate
