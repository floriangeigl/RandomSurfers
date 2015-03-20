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
import scipy
import pandas as pd
import operator
import random

np.set_printoptions(linewidth=225)
import seaborn
from timeit import Timer
np.set_printoptions(precision=2)

# np.set_printoptions(linewidth=1200, precision=1)

def calc_entropy(AT, norm_sigma_in=None, known_nodes=None):
    #print 'AT\n', AT.todense(), type(AT)
    #print 'norm_sigma_in in\n', norm_sigma_in, type(norm_sigma_in)
    #print '=' * 80
    selection_range = set(range(AT.shape[0]))
    if norm_sigma_in is None:
        norm_sigma_in = scipy.sparse.csr_matrix(np.ones(AT.shape, dtype=np.float))

    if known_nodes is not None:
        norm_sigma_in = scipy.sparse.csr_matrix(np.ones(AT.shape, dtype=np.float))
        #print 'ones:\n', norm_sigma_in.todense(), type(norm_sigma_in)
        #print 'orig norm_sigma_in:\n', norm_sigma_in, type(norm_sigma_in)
        for v in map(int, known_nodes):
            norm_sigma_in[v, :] = norm_sigma_in[v, :]
            norm_sigma_in[:, v] = norm_sigma_in[:, v]
        #print 'mixed norm_sigma_in at:', sorted(known_nodes),'\n',norm_sigma_in.todense()
    #print 'norm_sigma_in:\n', norm_sigma_in.todense(), type(norm_sigma_in)
    total_entropy = 0
    for v in selection_range:
        # exclude loop
        current_selection = list(selection_range - {v})
        # stack the katz row of the target vertex N-1 times
        sigma_row = norm_sigma_in[v, :]
        # multiply katz with transposed A -> only katz values on real links
        #print 'stacked norm_sigma_in:\n', sigma_row.todense()
        #print 'sliced AT:\n', AT[current_selection,:].todense(), type(AT[current_selection,:])
        if norm_sigma_in is not None:
            #print sigma_row.shape
            res = AT[current_selection, :].multiply(sigma_row)
            #print res.todense(), type(res)
        else:
            res = AT[current_selection, :]
        # calc entropy per row and add it to the overall entropy
        ent = stats.entropy(res.T.todense())
        #print ent
        # if np.isinf(ent.sum()):
        # print ent
        # print res.sum(axis=1)
        # exit()
        total_entropy += ent.sum()
    num_v = AT.shape[0]
    total_entropy /= (num_v * (num_v - 1))
    return total_entropy


def test_entropy(net, name='entropy_tests', out_dir='output/tests/', granularity=10, num_samples=20):
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
    A = adjacency(net)
    AT = A.T.tocsr()
    try:
        l, v = matrix_spectrum(A, sparse=True)
    except:
        l, v = matrix_spectrum(A, sparse=False)
    kappa_1 = l[0].real
    alpha_max = 1.0 / kappa_1
    alpha_max *= 0.99
    alpha = alpha_max
    sigma_global = katz_sim_matrix(A, alpha)
    normalized_sigma = sigma_global / sigma_global.mean(axis=0).T
    #sigma_global = scipy.sparse.csr_matrix(sigma_global)
    #sigma_global /= sigma_global.sum(axis=1)
    print 'calc baselines'
    adj_entropy = calc_entropy(AT)
    print '\tA entropy:', adj_entropy
    sigma_entropy = calc_entropy(AT, normalized_sigma)
    print '\tsigma entropy', sigma_entropy
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
        for j in range(num_samples):
            known_nodes = set(random.sample(vertice_range, i))
            if i > net.num_vertices() / 2:
                t = Timer(lambda: calc_entropy(AT, normalized_sigma, known_nodes=known_nodes))
                time = t.timeit(number=100)
                print 'calc entropy time:', time
                exit()
            entropies.append((calc_entropy(AT, normalized_sigma, known_nodes=known_nodes), len(known_nodes)))
    print 'plot'
    data, idx = map(list, zip(*sorted(entropies, key=operator.itemgetter(1))))
    df = pd.DataFrame(columns=['random tests'], data=data)
    df['num_nodes'] = idx
    df_max = df.groupby('num_nodes').describe()
    df = df_max.unstack().reset_index()
    df = pd.concat([df['num_nodes'], df['random tests']], axis=1)
    df['adj'] = adj_entropy
    df['sigma'] = sigma_entropy
    df.plot(x='num_nodes', y=['adj', 'sigma', 'mean', 'min', 'max'], lw=2,
            color=['black', 'green', 'blue', 'lightblue', 'darkblue'], alpha=0.7,
            label=['adj', 'sigma'])
    plt.xlabel('#known nodes')
    plt.ylabel('entropy (#random samples: ' + str(num_samples) + ')')
    # print 'min'.center(20, '=')
    # print df.min()
    # print 'max'.center(20, '=')
    #print df.max()
    plt.savefig(out_dir + name + '.png', bbox='tight')
    plt.close('all')


generator = SBMGenerator()
granularity = 10
num_samples = 10
outdir = './output/'

test = True

if test:
    print 'sbm'.center(80, '=')
    name = 'sbm_n10_m30'
    net = generator.gen_stock_blockmodel(num_nodes=10, blocks=1, num_links=20)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name, num_samples=num_samples)
    print 'price network'.center(80, '=')
    name = 'price_net_n50_m1_g2_1'
    net = price_network(30, m=1, gamma=1, directed=False)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name, num_samples=num_samples)
    print 'complete graph'.center(80, '=')
    name = 'complete_graph_n50'
    net = complete_graph(30)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name, num_samples=num_samples)
    print 'quick tests done'.center(80, '=')
else:
    num_links = 2000
    num_nodes = 1000
    num_blocks = 10
    print 'sbm'.center(80, '=')
    name = 'sbm_strong_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.1)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name,
                 num_samples=num_samples)
    print 'sbm'.center(80, '=')
    name = 'sbm_weak_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.7)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name,
                 num_samples=num_samples)
    print 'powerlaw'.center(80, '=')
    name = 'powerlaw_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=1, num_links=num_links)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name,
                 num_samples=num_samples)
    print 'price network'.center(80, '=')
    name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_vertices)
    net = price_network(num_nodes, m=2, gamma=1, directed=False)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name,
                 num_samples=num_samples)
    print 'complete graph'.center(80, '=')
    name = 'complete_graph_n' + str(num_nodes)
    net = complete_graph(num_nodes)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name, num_samples=num_samples)
    print 'karate'.center(80, '=')
    name = 'karate'
    net = load_edge_list('/opt/datasets/karate/karate.edgelist')
    generator.analyse_graph(net, outdir + name, draw_net=False)
    test_entropy(net, granularity=granularity, name=name)
    exit()
    print 'wiki4schools'.center(80, '=')
    net = load_edge_list('/opt/datasets/wikiforschools/graph')
    test_entropy(net, granularity=granularity, name='wiki4schools')
    print 'facebook'.center(80, '=')
    net = load_edge_list('/opt/datasets/facebook/facebook')
    test_entropy(net, granularity=granularity, name='facebook')
    print 'youtube'.center(80, '=')
    net = load_edge_list('/opt/datasets/youtube/youtube')
    test_entropy(net, granularity=granularity, name='youtube')
    print 'dblp'.center(80, '=')
    net = load_edge_list('/opt/datasets/dblp/dblp')
    test_entropy(net, granularity=granularity, name='dblp')




