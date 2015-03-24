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
import linalg as la
import scipy.linalg as lalg
import scipy.sparse.linalg as linalg
import scipy.stats as stats
import scipy
from scipy.sparse.csr import csr_matrix
import pandas as pd
import operator
import random
from collections import defaultdict
import multiprocessing
import datetime
import traceback
np.set_printoptions(linewidth=225)
import seaborn
from timeit import Timer
np.set_printoptions(precision=2)
import copy
import matplotlib.cm as colormap

def katz_sim_network(net,largest_eigenvalue=None, gamma=0.99):
    if largest_eigenvalue is None:
        largest_eigenvalue, _ = eigenvector(net)
    #kappa_1 = l[0].real
    alpha_max = 1.0 / largest_eigenvalue
    alpha = gamma * alpha_max
    katz = la.katz_matrix(adjacency(net), alpha)
    sigma = lalg.inv(katz)
    return sigma


def entropy_rate(M, stat_dist=None, base=2):
    if stat_dist is None:
        stat_dist = stationary_dist(M)
    if scipy.sparse.issparse(M):
        M = M.todense()
    return (stats.entropy(M.T, base=base) * stat_dist).sum()


def draw_graph(network, color, shape=None, sizep=None, colormap_name='spring', min_vertex_size_shrinking_factor=4,
               output='graph.png', output_size=2048, **kwargs):
    num_nodes = network.num_vertices()
    min_vertex_size_shrinking_factor = min_vertex_size_shrinking_factor
    if num_nodes < 10:
        num_nodes = 10
    max_vertex_size = np.sqrt((np.pi * (output_size / 2) ** 2) / num_nodes)
    if max_vertex_size < min_vertex_size_shrinking_factor:
        max_vertex_size = min_vertex_size_shrinking_factor
    min_vertex_size = max_vertex_size / min_vertex_size_shrinking_factor
    if sizep is None:
        sizep = max_vertex_size + min_vertex_size
        sizep /= 3
    else:
        sizep = prop_to_size(sizep, mi=min_vertex_size / 3 * 2, ma=max_vertex_size / 3 * 2, power=2)
    if shape is None:
        shape = 'circle'
    elif isinstance(shape, str):
        try:
            shape = network.vp[shape]
            shape.a %= 14
        except KeyError:
            shape = 'circle'
            print 'cannot find shape property:', shape

    output_size = (output_size, output_size)
    cmap = colormap.get_cmap(colormap_name)
    color = color.copy()
    try:
        _ = color.a
    except AttributeError:
        c = network.new_vertex_property('float')
        c.a = color
        color = c
    color.a -= color.a.mean()
    color.a /= color.a.var()
    color.a += 1
    color.a /= 2
    if not output.endswith('.png'):
        output += '.png'
    color_pmap = network.new_vertex_property('vector<float>')
    tmp = np.array([np.array(cmap(i)) for i in color.a])
    color_pmap.set_2d_array(tmp.T)
    graph_draw(network, vertex_fill_color=color_pmap, vertex_pen_width=0.0, vertex_shape=shape,
               bg_color=[1, 1, 1, 1], edge_color=[0.179, 0.203, 0.210, 0.1], vertex_size=sizep, output_size=output_size,
               output=output, **kwargs)
    plt.close('all')


def calc_entropy_and_stat_dist(A, M=None):
    if M is not None:
        if not A.shape == M.shape:
            M = np.diag(M)
        assert A.shape == M.shape
        weighted_trans = A.dot(M)
    else:
        weighted_trans = A.copy()
    weighted_trans = normalize_mat(weighted_trans)
    stat_dist = stationary_dist(weighted_trans)
    return entropy_rate(weighted_trans, stat_dist=stat_dist), stat_dist


def normalize_mat(M, copy=False):
    if copy:
        M = M.copy()
    return M / M.sum(axis=1)


def stationary_dist(M):
    M = normalize_mat(M, copy=True)
    return la.leading_eigenvector(M.T)[1]

def self_sim_entropy(network, name, out_dir):
    A = adjacency(network)
    A_eigvalue, A_eigvector = eigenvector(network)
    A_eigvector = A_eigvector.a
    weights = dict()
    weights['adjacency'] = None
    weights['eigenvector'] = A_eigvector
    weights['sigma'] = katz_sim_network(network, largest_eigenvalue=A_eigvalue)
    pos = None
    deg_map = network.degree_property_map('total')

    P = la.transition_matrix(A.todense())
    l, v = la.leading_eigenvector(P.T)


    entropy_df = pd.DataFrame()
    for key, weight in weights.iteritems():
        print key.center(80, '=')
        ent, stat_dist = calc_entropy_and_stat_dist(A.todense(), weight)
        print 'entropy rate:', ent
        entropy_df.at[0, key] = ent
        if False:
            if weight is not None and weight.shape != A.shape:
                weight = np.diag(weight)
            d_erate = la.rw_entropy_rate(np.array(A.todense())) if weight is None else la.rw_entropy_rate(
                np.dot(np.array(A.todense()), weight))
            print 'denis calc:', d_erate
            assert np.allclose(d_erate, ent)
            P = la.transition_matrix(np.dot(A.todense(), weight) if weight is not None else A.todense())
            l1, v1 = la.leading_eigenvector(P.T)
            print 'denis calc:', v1
            assert np.allclose(v1, stat_dist)
            print 'stationary dist:', stat_dist
        print 'draw graph:', out_dir + name + '_' + key
        if pos is None:
            pos = sfdp_layout(network)
        draw_graph(network, color=stat_dist, sizep=deg_map, shape='com', output=out_dir + name + '_' + key, pos=pos)
    print entropy_df
    entropy_df.plot(kind='bar')
    plt.ylabel('entropy rate')
    plt.savefig(out_dir + name + '_entropy_rates.png')
    plt.close('all')

#=======================================================================================================================


def main():
    generator = SBMGenerator()
    granularity = 10
    num_samples = 10
    outdir = 'output/'

    test = False

    if test:
        outdir += 'tests/'
        print 'sbm'.center(80, '=')
        name = 'sbm_n10_m30'
        net = generator.gen_stock_blockmodel(num_nodes=100, blocks=3, num_links=400, self_con=1, other_con=0.01)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net, name=name, out_dir=outdir)
        print 'price network'.center(80, '=')
        name = 'price_net_n50_m1_g2_1'
        net = price_network(30, m=1, gamma=1, directed=False)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net, name=name, out_dir=outdir)
        print 'complete graph'.center(80, '=')
        name = 'complete_graph_n50'
        net = complete_graph(30)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net, name=name, out_dir=outdir)
        print 'quick tests done'.center(80, '=')
    else:
        num_links = 2000
        num_nodes = 1000
        num_blocks = 5
        print 'sbm'.center(80, '=')
        name = 'sbm_strong_n' + str(num_nodes) + '_m' + str(num_links)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.05)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net, name=name, out_dir=outdir)
        print 'sbm'.center(80, '=')
        name = 'sbm_weak_n' + str(num_nodes) + '_m' + str(num_links)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.5)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net,name=name, out_dir=outdir)
        print 'powerlaw'.center(80, '=')
        name = 'powerlaw_n' + str(num_nodes) + '_m' + str(num_links)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=1, num_links=num_links)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net,  name=name, out_dir=outdir)
        print 'price network'.center(80, '=')
        name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_edges())
        net = price_network(num_nodes, m=2, gamma=1, directed=False)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net,  name=name, out_dir=outdir)
        print 'complete graph'.center(80, '=')
        name = 'complete_graph_n' + str(num_nodes)
        net = complete_graph(num_nodes)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net, name=name, out_dir=outdir)
        print 'karate'.center(80, '=')
        name = 'karate'
        net = load_edge_list('/opt/datasets/karate/karate.edgelist')
        generator.analyse_graph(net, outdir + name, draw_net=False)
        self_sim_entropy(net,  name=name, out_dir=outdir)
        exit()
        print 'wiki4schools'.center(80, '=')
        net = load_edge_list('/opt/datasets/wikiforschools/graph')
        self_sim_entropy(net,  name='wiki4schools')
        print 'facebook'.center(80, '=')
        net = load_edge_list('/opt/datasets/facebook/facebook')
        self_sim_entropy(net, name='facebook')
        print 'youtube'.center(80, '=')
        net = load_edge_list('/opt/datasets/youtube/youtube')
        self_sim_entropy(net,  name='youtube')
        print 'dblp'.center(80, '=')
        net = load_edge_list('/opt/datasets/dblp/dblp')
        self_sim_entropy(net,  name='dblp')


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start