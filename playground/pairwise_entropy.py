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
from collections import defaultdict
import multiprocessing
import traceback
np.set_printoptions(linewidth=225)
import seaborn
from timeit import Timer
np.set_printoptions(precision=2)
import copy

# np.set_printoptions(linewidth=1200, precision=1)


def calc_entropy_mp(AT, sigma, known_nodes):
    return calc_entropy(AT, sigma=sigma, known_nodes=known_nodes)


def calc_entropy(AT, sigma=None, weights=None, known_nodes=None, entropy_base=2):
    if weights is not None:
        weights = np.array(weights)
        if weights.sum() != 1.0:
            weights /= weights.sum()
    if sigma is None:
        entropy = stats.entropy(AT.T, base=entropy_base)
        if weights is not None:
            entropy *= weights
        return entropy.mean()

    selection_range = set(range(AT.shape[0]))
    # print 'sigma:\n', sigma
    if known_nodes is not None:
        ones_mat = np.ones(AT.shape, dtype=np.float)
        # sigma *= sigma_in.min()
        # print 'known nodes:', known_nodes
        #print 'orig sigma\n', sigma
        for v in map(int, known_nodes):
            ones_mat[v, :] = sigma[v, :]
            ones_mat[:, v] = sigma[:, v]
        sigma = ones_mat
        #sigma[0, 2] = 100
        #print 'sigma:\n', sigma
        #print 'max:', sigma.max(axis=1).reshape(sigma.shape[1], 1)
        sigma /= sigma.mean(axis=1).reshape(sigma.shape[1], 1)
        #print 'norm sigma:\n', sigma
    #print 'mean:', sigma.mean(axis=1)
    # print 'sigma:\n', sigma
    total_entropy = 0
    for v in selection_range:
        # exclude loop
        current_selection = list(selection_range - {v})
        # stack the katz row of the target vertex N-1 times
        #print sigma
        row_sigma = sigma[v, :]
        #print row_sigma
        #print 'AT\n',AT
        #print '@:',current_selection
        #print AT[:,current_selection]
        # multiply katz with transposed AT -> only katz values on real links
        res = np.multiply(row_sigma, AT[current_selection, :])
        #print res
        # calc entropy per row and add it to the overall entropy
        ent = stats.entropy(res.T, base=entropy_base)
        if weights is not None:
            ent *= weights[current_selection]
        #print ent
        total_entropy += ent.sum()
    num_v = AT.shape[0]
    total_entropy /= (num_v * (num_v - 1))
    print 'total entropy:', total_entropy

    #print 'total entropy:', total_entropy
    if known_nodes is not None:
        return total_entropy, int(len(known_nodes) / AT.shape[0] * 100)
    return total_entropy


def entropy_to_all_targets(net, name='entropy_tests', out_dir='output/tests/', granularity=10, num_samples=20):
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
    print 'calc network centrality'
    rankings = dict()
    ranking_res = defaultdict(list)
    print '\tdegree'
    pmap = net.degree_property_map('total')
    rankings['deg'] = map(int, sorted(net.vertices(), key=lambda x: pmap[x], reverse=True))

    print '\tpage rank'
    pmap = pagerank(net)
    rankings['page rank'] = map(int, sorted(net.vertices(), key=lambda x: pmap[x], reverse=True))

    print '\tbetweenness'
    pmap, _ = betweenness(net)
    rankings['betweenness'] = map(int, sorted(net.vertices(), key=lambda x: pmap[x], reverse=True))

    print '\tkatz'
    pmap = katz(net)
    rankings['katz'] = map(int, sorted(net.vertices(), key=lambda x: pmap[x], reverse=True))

    print '\teigenvector'
    largest_eigen_vector, pmap = eigenvector(net, max_iter=2000)
    rankings['eigenvector'] = map(int, sorted(net.vertices(), key=lambda x: pmap[x], reverse=True))

    print '\tclustering'
    pmap = local_clustering(net)
    rankings['clustering'] = map(int, sorted(net.vertices(), key=lambda x: pmap[x], reverse=True))

    print 'calc katz similarity'
    A = adjacency(net)
    AT = A.T.todense()
    #try:
    #    l, v = matrix_spectrum(A, sparse=True)
    #except:
    #    l, v = matrix_spectrum(A, sparse=False)
    #kappa_1 = l[0].real
    alpha_max = 1.0 / largest_eigen_vector
    alpha_max *= 0.99
    alpha = alpha_max
    sigma_global = katz_sim_matrix(A, alpha)
    _ = calc_katz_iterative(A, alpha, max_iter=2000, filename=name + '_katz_range', out_dir=out_dir, plot=True)
    norm_sigma = sigma_global / sigma_global.mean(axis=1).reshape(sigma_global.shape[1], 1)

    print 'calc baselines'
    adj_entropy = calc_entropy(AT)
    print '\tA entropy:', adj_entropy
    sigma_entropy = calc_entropy(AT, norm_sigma)
    print '\tsigma entropy', sigma_entropy
    entropies = list()
    print 'start tests with granularity', granularity, 'a', num_samples, 'samples'
    step_size = (net.num_vertices() / granularity)
    vertice_range = set(range(A.shape[0]))
    worker_pool = multiprocessing.Pool(processes=12)
    for num_known_nodes in range(granularity + 1):
        print '.',
        num_known_nodes *= step_size
        num_known_nodes = int(num_known_nodes)
        for j in range(num_samples):
            known_nodes = set(random.sample(vertice_range, num_known_nodes))
            if False and num_known_nodes > net.num_vertices() / 2:
                t = Timer(lambda: calc_entropy(AT, norm_sigma, known_nodes=known_nodes))
                time = t.timeit(number=100)
                print 'calc entropy time:', time
                exit()
            worker_pool.apply_async(calc_entropy_mp, (AT, norm_sigma, known_nodes,), callback=entropies.append)
            # entropies.append((calc_entropy(AT, norm_sigma, known_nodes=known_nodes), len(known_nodes)))
        for key, val in rankings.iteritems():
            ranking_res[key].append(calc_entropy(AT, norm_sigma, known_nodes=val[:num_known_nodes])[0])
    worker_pool.close()
    worker_pool.join()
    print 'plot'
    data, idx = map(list, zip(*sorted(entropies, key=operator.itemgetter(1))))
    df = pd.DataFrame(columns=['random tests'], data=data)
    df['num_nodes'] = idx
    df_max = df.groupby('num_nodes').describe()
    df = df_max.unstack().reset_index()
    df = pd.concat([df['num_nodes'], df['random tests']], axis=1)
    df['adj'] = adj_entropy
    df['sigma'] = sigma_entropy
    for key, val in ranking_res.iteritems():
        df[key] = val
    ax = df.plot(x='num_nodes', y=['adj', 'sigma', 'mean'], lw=1,
                 color=['black', 'green', 'blue', 'lightblue', 'darkblue'], alpha=0.3)
    df.plot(x='num_nodes', y=ranking_res.keys(), lw=3,
            color=['black', 'green', 'blue', 'lightblue', 'darkblue'], alpha=0.7, ax=ax)
    plt.xlabel('known nodes in %')
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
    outdir += 'tests/'
    print 'sbm'.center(80, '=')
    name = 'sbm_n10_m30'
    net = generator.gen_stoch_blockmodel(num_nodes=100, blocks=3, num_links=200)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    print 'price network'.center(80, '=')
    name = 'price_net_n50_m1_g2_1'
    net = price_network(30, m=1, gamma=1, directed=False)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    print 'complete graph'.center(80, '=')
    name = 'complete_graph_n50'
    net = complete_graph(30)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    print 'quick tests done'.center(80, '=')
else:
    num_links = 2000
    num_nodes = 1000
    num_blocks = 10
    print 'sbm'.center(80, '=')
    name = 'sbm_strong_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stoch_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.1)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    print 'sbm'.center(80, '=')
    name = 'sbm_weak_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stoch_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.7)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    print 'powerlaw'.center(80, '=')
    name = 'powerlaw_n' + str(num_nodes) + '_m' + str(num_links)
    net = generator.gen_stoch_blockmodel(num_nodes=num_nodes, blocks=1, num_links=num_links)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    print 'price network'.center(80, '=')
    name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_edges())
    net = price_network(num_nodes, m=2, gamma=1, directed=False)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    print 'complete graph'.center(80, '=')
    name = 'complete_graph_n' + str(num_nodes)
    net = complete_graph(num_nodes)
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    print 'karate'.center(80, '=')
    name = 'karate'
    net = load_edge_list('/opt/datasets/karate/karate.edgelist')
    generator.analyse_graph(net, outdir + name, draw_net=False)
    entropy_to_all_targets(net, granularity=granularity, name=name, out_dir=outdir, num_samples=num_samples)
    exit()
    print 'wiki4schools'.center(80, '=')
    net = load_edge_list('/opt/datasets/wikiforschools/graph')
    entropy_to_all_targets(net, granularity=granularity, name='wiki4schools')
    print 'facebook'.center(80, '=')
    net = load_edge_list('/opt/datasets/facebook/facebook')
    entropy_to_all_targets(net, granularity=granularity, name='facebook')
    print 'youtube'.center(80, '=')
    net = load_edge_list('/opt/datasets/youtube/youtube')
    entropy_to_all_targets(net, granularity=granularity, name='youtube')
    print 'dblp'.center(80, '=')
    net = load_edge_list('/opt/datasets/dblp/dblp')
    entropy_to_all_targets(net, granularity=granularity, name='dblp')




