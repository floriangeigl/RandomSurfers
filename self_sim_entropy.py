from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from graph_tool.all import *
import matplotlib.pylab as plt

import sys
import os
from tools.gt_tools import SBMGenerator, load_edge_list, load_property
import tools.basics as basics
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
               output='graph.png', output_size=800,standardize=False, **kwargs):
    print 'draw graph ||',
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
            print 'cannot find shape property:', shape, '||',
            shape = 'circle'

    output_size = (output_size, output_size)
    cmap = colormap.get_cmap(colormap_name)
    color = color.copy()
    try:
        _ = color.a
    except AttributeError:
        c = network.new_vertex_property('float')
        c.a = color
        color = c
    if standardize:
        color.a -= color.a.mean()
        color.a /= color.a.var()
        color.a += 1
        color.a /= 2
    else:
        color.a -= color.a.min()
        color.a /= color.a.max()
    if not output.endswith('.png'):
        output += '.png'
    color_pmap = network.new_vertex_property('vector<float>')
    tmp = np.array([np.array(cmap(i)) for i in color.a])
    color_pmap.set_2d_array(tmp.T)
    graph_draw(network, vertex_fill_color=color_pmap, vertex_pen_width=0.0, vertex_shape=shape,
               bg_color=[1, 1, 1, 1], edge_color=[0.179, 0.203, 0.210, 0.1], vertex_size=sizep, output_size=output_size,
               output=output, **kwargs)
    plt.close('all')
    print 'done'


def calc_entropy_and_stat_dist(A, M=None):
    if M is not None:
        #if A.shape != M.shape:
        #    M = np.diag(M)
        #    assert A.shape == M.shape
        #    weighted_trans = A.dot(M)
        #else:
        weighted_trans = A.multiply(M)
    else:
        weighted_trans = A.copy()
    weighted_trans = normalize_mat(weighted_trans)
    stat_dist = stationary_dist(weighted_trans)
    return entropy_rate(weighted_trans, stat_dist=stat_dist), stat_dist


def normalize_mat(M, copy=False, replace_nans_with=0):
    if copy:
        M = M.copy()
    M /= M.sum(axis=1)
    if replace_nans_with is not None:
        sum = M.sum()
        if np.isnan(sum) or np.isinf(sum):
            print 'warn replacing nans with zero'
            M[np.invert(np.isfinite(M))] = replace_nans_with
    return M


def stationary_dist(M):
    M = normalize_mat(M, copy=True)
    return la.leading_eigenvector(M.T)[1]


def calc_cosine(A, weight_direct_link=False):
    if weight_direct_link:
        A = A.copy() + np.eye(A.shape[0])
    com_neigh = A.dot(A)
    deg = A.sum(axis=1).astype('float')
    deg_norm = np.sqrt(deg * deg.T)
    com_neigh /= deg_norm
    return com_neigh


def self_sim_entropy(network, name, out_dir):
    A = adjacency(network)
    A_eigvalue, A_eigvector = eigenvector(network)
    A_eigvector = A_eigvector.a
    deg_map = network.degree_property_map('total')
    weights = dict()
    weights['adjacency'] = None
    weights['eigenvector'] = A_eigvector
    weights['sigma'] = katz_sim_network(network, largest_eigenvalue=A_eigvalue)
    weights['deg_corrected_sigma'] = weights['sigma'] / deg_map.a
    weights['log_deg_corrected_sigma'] = weights['sigma'] / np.log(deg_map.a)
    weights['pagerank'] = pagerank(network).a
    weights['betweenness'] = betweenness(network)[0].a
    weights['katz'] = katz(network).a
    weights['cosine'] = calc_cosine(A)
    weights['cosine_direct_links'] = calc_cosine(A, weight_direct_link=True)


    entropy_df = pd.DataFrame()
    print 'calc graph-layout'
    pos = sfdp_layout(network)
    for key, weight in weights.iteritems():
        print key.center(80, '=')
        ent, stat_dist = calc_entropy_and_stat_dist(A, weight)
        print 'entropy rate:', ent
        entropy_df.at[0, key] = ent
        stat_dist_ser = pd.Series(data=stat_dist)
        stat_dist_ser.plot(kind='hist', bins=100, lw=0, normed=True)
        plt.title(key)
        plt.ylabel('#nodes')
        plt.xlabel('stationary value')
        plt.savefig(out_dir + name + '_stat_dist_' + key + '.png', bbox_tight=True)
        plt.close('all')
        #print 'draw graph:', out_dir + name + '_' + key
        draw_graph(network, color=stat_dist, sizep=deg_map, shape='com', output=out_dir + name + '_graph_' + key,
                   pos=pos)
    entropy_df.sort(axis=1, inplace=True)
    print entropy_df
    ax = entropy_df.plot(kind='bar', label=sorted([i.replace('_', ' ') for i in entropy_df.columns]))
    min_e, max_e = entropy_df.loc[0].min(), entropy_df.loc[0].max()
    ax.set_ylim([min_e * 0.99, max_e * 1.01])
    plt.ylabel('entropy rate')
    plt.legend(loc='upper left')
    plt.savefig(out_dir + name + '_entropy_rates.png')
    plt.close('all')

#=======================================================================================================================


def main():
    generator = SBMGenerator()
    granularity = 10
    num_samples = 10
    outdir = 'output/'
    basics.create_folder_structure(outdir)

    test = False
    multip = True
    worker_pool = multiprocessing.Pool(processes=14)
    if test:
        outdir += 'tests/'
        basics.create_folder_structure(outdir)

        print 'sbm'.center(80, '=')
        name = 'sbm_n10_m30'
        net = generator.gen_stock_blockmodel(num_nodes=10, blocks=2, num_links=40, self_con=1, other_con=0.1)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        print 'price network'.center(80, '=')
        name = 'price_net_n50_m1_g2_1'
        net = price_network(30, m=2, gamma=1, directed=False)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        print 'complete graph'.center(80, '=')
        name = 'complete_graph_n50'
        net = complete_graph(30)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
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
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        print 'sbm'.center(80, '=')
        name = 'sbm_weak_n' + str(num_nodes) + '_m' + str(num_links)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.5)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        print 'powerlaw'.center(80, '=')
        name = 'powerlaw_n' + str(num_nodes) + '_m' + str(num_links)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=1, num_links=num_links)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        print 'price network'.center(80, '=')
        name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_edges())
        net = price_network(num_nodes, m=2, gamma=1, directed=False)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        print 'complete graph'.center(80, '=')
        name = 'complete_graph_n' + str(num_nodes)
        net = complete_graph(num_nodes)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        print 'karate'.center(80, '=')
        name = 'karate'
        net = load_edge_list('/opt/datasets/karate/karate.edgelist')
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        if True:
            print 'wiki4schools'.center(80, '=')
            name = 'wiki4schools'
            net = load_edge_list('/opt/datasets/wikiforschools/graph')
            net['com'] = load_property(net, '/opt/datasets/wikiforschools/artid_catid', type='int')
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)
            print 'facebook'.center(80, '=')
            name = 'facebook'
            net = load_edge_list('/opt/datasets/facebook/facebook')
            net['com'] = load_property(net, '/opt/datasets/facebook/facebook_com', type='int')
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)
            print 'youtube'.center(80, '=')
            name = 'youtube'
            net = load_edge_list('/opt/datasets/youtube/youtube')
            net['com'] = load_property(net, '/opt/datasets/youtube/youtube_com', type='int')
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)
            print 'dblp'.center(80, '=')
            name = 'dblp'
            net = load_edge_list('/opt/datasets/dblp/dblp')
            net['com'] = load_property(net, '/opt/datasets/dblp/dblp_com', type='int')
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)

    if multip:
        worker_pool.close()
        worker_pool.join()


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start