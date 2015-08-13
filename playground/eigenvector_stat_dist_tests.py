from __future__ import division
import graph_tool.all as gt
from playground.random_surfer import RandomSurfer
import random
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import eigs
from scipy.sparse import diags

def check_aperiodic(g):
    a = gt.adjacency(g)
    b = a * a
    diag_two_sum = b.diagonal().sum()
    # print '\tA*A diag sum:', int(diag_two_sum)
    b *= a
    diag_three_sum = b.diagonal().sum()
    return bool(diag_two_sum) and bool(diag_three_sum)

deg_samplers = ['unif', 'exp']
num_nodes = 20
iterations = 100
processes = 12
directed = True

spacing = 15
print 'nodes | iteration | deg sampler | bias'.rjust(20), '|', 'eigenvec status'.rjust(
    spacing), '|', 'compare status'.rjust(spacing), '| abstol', '| maxdiff'
print '#' * 120
for iteration in range(iterations):
    deg_sample_type = random.sample(deg_samplers, 1)[0]
    if deg_sample_type == 'unif':
        helper = lambda: int(random.random() * num_nodes)
    elif deg_sample_type == 'exp':
        helper = lambda: np.random.exponential(scale=0.06) * num_nodes
    deg_sample = lambda: (helper(), helper())
    # print deg_sample()
    while True:
        g = gt.random_graph(num_nodes, deg_sampler=deg_sample, directed=directed)
        use_weights = random.random() > 0.5
        if use_weights:
            weights = g.new_vertex_property('float')
            weights.a = np.array(
                [(1. + random.random() * 9) if i else 1. for i in (np.random.random(size=g.num_vertices()) > 0.5)])
        else:
            weights = None

        lcc = gt.label_largest_component(g)
        g.set_vertex_filter(lcc)
        g.purge_vertices()
        g.purge_edges()
        g.clear_filters()
        if g.num_vertices() > 5 and check_aperiodic(g):
            break
    print str(str(g.num_vertices()).ljust(5) + ' | ' + str(iteration).ljust(4) + ' | ' + deg_sample_type.ljust(5) + ' | ' + str(
            use_weights).ljust(6)).ljust(20),

    # eigenvector stat dist
    A = gt.adjacency(g)
    if weights is not None:
        bias = diags(weights.a, 0)
        A = bias.dot(A)
    Q = normalize(A, norm='l1', axis=0, copy=False)
    eig_val, eig_vec = eigs(Q, k=1, which="LR")
    eig_val = eig_val.real
    eig_vec = eig_vec[:, 0].real
    eig_vec_stat = eig_vec / eig_vec.sum()
    if np.isclose(eig_vec_stat.sum(), 1.) and np.allclose(eig_vec, Q * eig_vec):
        print '[OK]'.center(spacing),
    else:
        print '[FAIL]'.center(spacing),

    abstol = 1/g.num_vertices()
    # power iteration method
    P = np.matrix(Q.T.todense())
    M = P.copy()
    for i in range(1000):
        M = M * P
    p_eig_vec = np.array(M[0, :]).flatten()
    if np.allclose(p_eig_vec, eig_vec_stat):
        print '[OK]'.center(spacing),
    else:
        print '[FAIL]'.center(spacing), "%.4f" % abstol, "%.6f" % np.absolute((p_eig_vec - eig_vec_stat)).max()
        print
        print p_eig_vec
        print eig_vec_stat



    # simulation stat dist
    rand_surfer = RandomSurfer(g, weights=weights)
    sim_stat_dist = rand_surfer.surf(processes=processes)

    # compare
    # print 'atol:', 1./g.num_vertices()

    if np.allclose(sim_stat_dist, eig_vec_stat, atol=abstol):
        print '[OK]'.center(spacing), "%.4f" % abstol, "%.6f" % np.absolute((sim_stat_dist - eig_vec_stat)).max(),
    else:
        print '[FAIL]'.center(spacing), abstol
        print
        print sim_stat_dist
        print eig_vec_stat
    print
