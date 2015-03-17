from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from graph_tool.all import *
import matplotlib.pylab as plt

import sys
from tools.gt_tools import SBMGenerator, load_edge_list
import numpy as np
from linalg import *
import scipy.stats as stats
import pandas as pd
import operator
import random

# np.set_printoptions(linewidth=1200, precision=1)

def calc_entropy(A, sigma_in=None, known_nodes=None):
    if sigma_in is None:
        sigma_in = np.ones(A.shape, dtype=np.float)
    AT = A.T
    selection_range = set(range(AT.shape[0]))
    if known_nodes is None:
        sigma = sigma_in
        sigma /= sigma.sum(axis=1)
    else:
        sigma = np.ones(A.shape, dtype=np.float)
        sigma /= sigma.sum(axis=1)
        # sigma *= sigma_in.min()
        # print 'known nodes:', known_nodes
        for v in map(int, known_nodes):
            sigma[v, :] = sigma_in[v, :]
            sigma[:, v] = sigma_in[:, v]
    # print 'sigma:\n', sigma
    total_entropy = 0
    for v in selection_range:
        # exclude loop
        current_selection = list(selection_range - {v})
        # stack the katz row of the target vertex N-1 times
        stacked_row_sigma = sigma[[v] * (sigma.shape[0] - 1), :]
        # multiply katz with transposed A -> only katz values on real links
        res = np.multiply(stacked_row_sigma, AT[current_selection, :])
        # calc entropy per row and add it to the overall entropy
        ent = stats.entropy(res.T)
        # if np.isinf(ent.sum()):
        # print ent
        # print res.sum(axis=1)
        #    exit()
        total_entropy += ent.sum()
    num_v = A.shape[0]
    total_entropy = total_entropy / (num_v * (num_v - 1))
    return total_entropy


def test_entropy(net, name='entropy_tests', out_dir='output/tests/', granularity=10):
    print net
    print 'draw network'
    deg_map = net.degree_property_map('total')
    try:
        fill_color = net.vp['com']
    except KeyError:
        fill_color = 'blue'
    graph_draw(net, vertex_size=prop_to_size(deg_map, mi=2, ma=15, power=1), vertex_fill_color=fill_color,
               output=out_dir + name + '_net.png', bg_color=[1, 1, 1, 1])
    plt.close('all')
    print 'calc katz'
    A = adjacency(net).todense()
    A /= A.sum(axis=1)
    try:
        l, v = matrix_spectrum(A, sparse=True)
    except:
        l, v = matrix_spectrum(A, sparse=False)
    kappa_1 = l[0].real
    alpha_max = 1.0 / kappa_1
    alpha_max *= 0.99
    alpha = alpha_max
    sigma_global = katz_sim_matrix(A, alpha)
    #sigma_global = np.multiply(sigma_global, A.T)
    sigma_global /= sigma_global.sum(axis=1)
    print 'calc baselines'
    adj_entropy = calc_entropy(A)
    print 'A entropy:', adj_entropy
    sigma_entropy = calc_entropy(A, sigma_global)
    print 'sigma entropy', sigma_entropy
    entropies = list()
    print 'start tests'
    step_size = (net.num_vertices() / granularity)
    vertice_range = set(range(A.shape[0]))
    for i in range(granularity + 1):
        print '.',
        i *= step_size
        i = int(i)
        if i % 100 == 99:
            print '+', 100 * int((i + 1) / 100), '\n'
        sys.stdout.flush()
        known_nodes = set(random.sample(vertice_range, i))
        entropies.append((calc_entropy(A, sigma_global, known_nodes=known_nodes), len(known_nodes)))
    print 'plot'
    data, idx = map(list, zip(*sorted(entropies, key=operator.itemgetter(1))))
    df = pd.DataFrame(columns=['random tests'], data=data, index=idx)
    df['adj'] = adj_entropy
    df['sigma'] = sigma_entropy
    df.plot(lw=2, color=['black', 'blue', 'green'], alpha=0.7)
    # print 'min'.center(20, '=')
    # print df.min()
    #print 'max'.center(20, '=')
    #print df.max()
    plt.savefig(out_dir + name + '.png')
    plt.close('all')


generator = SBMGenerator()
granularity = 10
print 'sbm'.center(80, '=')
net = generator.gen_stock_blockmodel(num_nodes=300, blocks=3, num_links=500)
test_entropy(net, granularity=granularity, name='sbm_n300_m500')
print 'sbm'.center(80, '=')
net = generator.gen_stock_blockmodel(num_nodes=500, blocks=5, num_links=2000)
test_entropy(net, granularity=granularity, name='sbm_n500_m2000')
print 'price network'.center(80, '=')
net = price_network(300, m=2, gamma=1, directed=False)
test_entropy(net, granularity=granularity, name='price_net_n300_m2_g2_1')
print 'price network'.center(80, '=')
net = price_network(300, m=1, gamma=1, directed=False)
test_entropy(net, granularity=granularity, name='price_net_n300_m1_g2_1')
print 'complete graph'.center(80, '=')
net = complete_graph(300)
test_entropy(net, granularity=granularity, name='complete_graph_n300')
print 'circular graph'.center(80, '=')
net = circular_graph(300, k=2, directed=False)
test_entropy(net, granularity=granularity, name='circular_graph_n300')
print 'facebook'.center(80, '=')
net = load_edge_list('/opt/datasets/facebook/facebook')
test_entropy(net, granularity=granularity, name='facebook')
print 'wiki4schools'.center(80, '=')
net = load_edge_list('/opt/datasets/wikiforschools/graph')
test_entropy(net, granularity=granularity, name='wiki4schools')
print 'dblp'.center(80, '=')
net = load_edge_list('/opt/datasets/dblp/dblp')
test_entropy(net, granularity=granularity, name='dblp')
print 'youtube'.center(80, '=')
net = load_edge_list('/opt/datasets/youtube/youtube')
test_entropy(net, granularity=granularity, name='youtube')
print 'karate'.center(80, '=')
net = load_edge_list('/opt/datasets/karate/karate.edgelist')
test_entropy(net, granularity=granularity, name='karate')


