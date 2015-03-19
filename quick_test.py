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

np.set_printoptions(linewidth=225)
import seaborn
from timeit import Timer
# np.set_printoptions(precision=2)

# np.set_printoptions(linewidth=1200, precision=1)

def calc_entropy(A, sigma_in=None, known_nodes=None):
    if sigma_in is None:
        sigma_in = np.ones(A.shape, dtype=np.float)
    AT = A.T
    selection_range = set(range(AT.shape[0]))
    if known_nodes is None:
        sigma = sigma_in
        sigma /= sigma.mean(axis=1)
    else:
        sigma = np.ones(A.shape, dtype=np.float)
        sigma /= sigma.mean(axis=1)
        sigma_in = sigma_in.copy()
        sigma_in /= sigma_in.mean(axis=1)
        # sigma *= sigma_in.min()
        # print 'known nodes:', known_nodes
        for v in map(int, known_nodes):
            sigma[v, :] = sigma_in[v, :]
            sigma[:, v] = sigma_in[:, v]
    sigma /= sigma.sum(axis=1)
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
        # exit()
        total_entropy += ent.sum()
    num_v = A.shape[0]
    total_entropy = total_entropy / (num_v * (num_v - 1))
    return total_entropy


def calc_katz_iterative(A, alpha, max_iter=2000, filename='katz_range', out_dir='output/tests/', plot=True):
    print 'calc katz iterative'
    print 'alpha:', alpha
    sigma = np.identity(A.shape[0])
    A_max, alphas = list(), list()
    orig_A = A.copy()
    orig_alpha = alpha
    for i in range(1, max_iter):
        if i > 1:
            A *= orig_A
            alpha *= orig_alpha
        M = np.multiply(A, alpha)
        sigma += M
        A_max.append(M.max())
        alphas.append(alpha)
        if np.allclose(A_max[-1], 0):
            print '\tbreak after length:', i
            break
    if plot:
        df = pd.DataFrame(columns=['max matrix value'], data=A_max)
        df['alpha'] = alphas
        df.plot(secondary_y=['alpha'], alpha=0.75, lw=2)
        plt.xlabel('path length')
        plt.ylabel('value')
        plt.savefig(out_dir + filename + '.png', bbox='tight')
        plt.close('all')
    return sigma


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
    sigma_global_iter = calc_katz_iterative(A, alpha, filename='katz_range' + name, out_dir=out_dir)
    if False:
        t = Timer(lambda: katz_sim_matrix(A, alpha))
        inverse = t.timeit(number=10)
        t = Timer(lambda: calc_katz_iterative(A, alpha, filename='katz_range' + name, out_dir=out_dir))
        iterative = t.timeit(number=10)
        print 'iterative:', iterative
        print 'inverse:', inverse
        exit()
    if False:
        sigma_global = katz_sim_matrix(A, alpha)
        print 'inverse calc:'
        print sigma_global
        print 'iterative calc:'
        print sigma_global_iter
        exit()
    sigma_global = sigma_global_iter
    #sigma_global /= sigma_global.sum(axis=1)
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
        for j in range(num_samples):
            known_nodes = set(random.sample(vertice_range, i))
            entropies.append((calc_entropy(A, sigma_global, known_nodes=known_nodes), len(known_nodes)))
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
outdir = 'output/'

test = True

if test:
    print 'sbm'.center(80, '=')
    name = 'sbm_n10_m30'
    net = generator.gen_stock_blockmodel(num_nodes=1000, blocks=10, num_links=2000)
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




