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


def draw_graph(network, color, min_color=None, max_color=None, groups=None, sizep=None, colormap_name='bwr',
               min_vertex_size_shrinking_factor=4,
               output='graph.png', output_size=(15, 15), dpi=80, standardize=False,color_bar=True, **kwargs):
    print 'draw graph ||',
    num_nodes = network.num_vertices()
    min_vertex_size_shrinking_factor = min_vertex_size_shrinking_factor
    if num_nodes < 10:
        num_nodes = 10
    max_vertex_size = np.sqrt((np.pi * (min(output_size) * dpi / 2) ** 2) / num_nodes)
    if max_vertex_size < min_vertex_size_shrinking_factor:
        max_vertex_size = min_vertex_size_shrinking_factor
    min_vertex_size = max_vertex_size / min_vertex_size_shrinking_factor
    if sizep is None:
        sizep = max_vertex_size + min_vertex_size
        sizep /= 3
    else:
        sizep = prop_to_size(sizep, mi=min_vertex_size / 3 * 2, ma=max_vertex_size / 3 * 2, power=2)
    v_shape = 'circle'
    if isinstance(groups, str):
        try:
            v_shape = network.vp[groups].copy()
            #groups = network.vp[groups]
            #unique_groups = set(np.array(groups.a))
            #num_groups = len(unique_groups)
            #groups_c_map = colormap.get_cmap('gist_rainbow')
            #groups_c_map = {i: groups_c_map(idx / (num_groups - 1)) for idx, i in enumerate(unique_groups)}
            #v_pen_color = network.new_vertex_property('vector<float>')
            #for v in network.vertices():
            #    v_pen_color = groups_c_map[groups[v]]

            v_shape.a %= 14
        except KeyError:
            print 'cannot find groups property:', groups, '||',
            v_shape = 'circle'

    cmap = colormap.get_cmap(colormap_name)
    color = color.copy()


    try:
        _ = color.a
    except AttributeError:
        c = network.new_vertex_property('float')
        c.a = color
        color = c
    min_color = color.a.min() if min_color is None else min_color
    max_color = color.a.max() if max_color is None else max_color

    #orig_color = np.array(color.a)
    if standardize:
        color.a -= color.a.mean()
        color.a /= color.a.var()
        color.a += 1
        color.a /= 2
    else:
        #color.a -= min_color
        #color.a /= max_color
        tmp = np.array(color.a)
        tmp[tmp > 100] = 100 + (tmp[tmp > 100] / (max_color/100))
        color.a = tmp
        color.a /= 200
    if not output.endswith('.png'):
        output += '.png'
    color_pmap = network.new_vertex_property('vector<float>')
    tmp = np.array([np.array(cmap(i)) for i in color.a])
    color_pmap.set_2d_array(tmp.T)
    plt.switch_backend('cairo')
    f, ax = plt.subplots(figsize=(15, 15))
    output_size = (output_size[0] * 0.8, output_size[1])
    edge_alpha = 0.3 if network.num_vertices() < 1000 else 0.01
    pen_width = 0.8 if network.num_vertices() < 1000 else 0.1
    v_pen_color = [0., 0., 0., 1] if network.num_vertices() < 1000 else [0.0, 0.0, 0.0, edge_alpha]
    graph_draw(network, vertex_fill_color=color_pmap, mplfig=ax, vertex_pen_width=pen_width, vertex_shape=v_shape,
               vertex_color=v_pen_color, edge_color=[0.179, 0.203, 0.210, edge_alpha], vertex_size=sizep,
               output_size=output_size, output=output, **kwargs)
    if color_bar:
        cmap = plt.cm.ScalarMappable(cmap=cmap)
        cmap.set_array([0, 200])
        cbar = f.colorbar(cmap, drawedges=False)
        ticks = [0, 1.0, max_color / 100]
        cbar.set_ticks([0, 100.0, 200.0])
        tick_labels = None
        non_zero_dig = 1
        for digi in range(10):
            tick_labels = [str("{:2." + str(digi) + "f}").format(i) for i in ticks]
            if any([len(i.replace('.', '').replace('0', '').replace(' ', '').replace('-', '').replace('+', '')) > 0 for
                    i in tick_labels]):
                non_zero_dig -= 1
                if non_zero_dig == 0:
                    break
        cbar.ax.set_yticklabels(tick_labels)
        #var = stats.tvar(orig_color)
        #cbar.set_label('')
    plt.axis('off')
    plt.savefig(output, bbox_inches='tight', dpi=dpi)
    plt.close('all')
    plt.switch_backend('Agg')
    print 'done'


def create_scatter(x, y, fname, c=None, cmap='blue'):
    assert isinstance(x, tuple)
    assert isinstance(y, tuple)
    df = pd.DataFrame(columns=[x[0]], data=x[1])
    df[y[0]] = y[1]
    if c is not None:
        # df['color'] = c
        pass
    df.plot(x=x[0], y=y[0], kind='scatter',
            alpha=1 / np.log10(len(df)), loglog=True, color='darkblue', lw=0)
    plt.axhline(1., color='red', alpha=.25, lw=2, ls='--')
    plt.savefig(fname, bbox_inches='tight')
    plt.close('all')



def katz_sim_network(net, largest_eigenvalue=None, gamma=0.99):
    if largest_eigenvalue is None:
        largest_eigenvalue, _ = eigenvector(net)
    # kappa_1 = l[0].real
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
    entropy_rate = np.nansum((stats.entropy(M.T, base=base) * stat_dist))
    assert np.isfinite(entropy_rate)
    return entropy_rate

def calc_entropy_and_stat_dist(A, M=None):
    if M is not None:
        if np.count_nonzero(M) == 0:
            print '\tall zero matrix as weights -> use ones-matrix'
            M = np.ones(M.shape, dtype='float')
        weighted_trans = A.multiply(M)
    else:
        weighted_trans = A.copy()
    weighted_trans = normalize_mat(weighted_trans)
    stat_dist = stationary_dist(weighted_trans)
    print 'entropy rate'
    return entropy_rate(weighted_trans, stat_dist=stat_dist), stat_dist


def normalize_mat(M, copy=False, replace_nans_with=0):
    if copy:
        M = M.copy()
    if np.count_nonzero(M) == 0:
        print '\tnormalize all zero matrix -> set to all 1 before normalization'
        M = np.ones(M.shape, dtype='float')
        # np.fill_diagonal(M, 0)
    M /= M.sum(axis=1)
    if replace_nans_with is not None:
        sum = M.sum()
        if np.isnan(sum) or np.isinf(sum):
            print 'warn replacing nans with zero'
            M[np.invert(np.isfinite(M))] = replace_nans_with
    assert np.all(np.isfinite(M))
    return M


def stationary_dist(M):
    M = normalize_mat(M, copy=True)
    stat_dist = la.leading_eigenvector(M.T)[1]
    assert np.all(np.isfinite(stat_dist))
    return stat_dist


def calc_cosine(A, weight_direct_link=False):
    if weight_direct_link:
        A = A.copy() + np.eye(A.shape[0])
    com_neigh = A.dot(A)
    deg = A.sum(axis=1).astype('float')
    deg_norm = np.sqrt(deg * deg.T)
    com_neigh /= deg_norm
    assert np.all(np.isfinite(com_neigh))
    return com_neigh


def calc_common_neigh(A):
    M = A.dot(A).todense()
    np.fill_diagonal(M, 0)
    return M

def self_sim_entropy(network, name, out_dir):
    A = adjacency(network)
    A_eigvalue, A_eigvector = eigenvector(network)
    A_eigvector = A_eigvector.a
    deg_map = network.degree_property_map('total')
    weights = dict()
    weights['adjacency'] = None
    weights['eigenvector'] = A_eigvector
    weights['eigenvector_inverse'] = 1 / A_eigvector
    weights['sigma'] = katz_sim_network(network, largest_eigenvalue=A_eigvalue)
    weights['sigma_deg_corrected'] = weights['sigma'] / np.array(deg_map.a)
    weights['cosine'] = calc_cosine(A, weight_direct_link=True)
    weights['betweenness'] = np.array(betweenness(network)[0].a)
    #weights['sigma_log_deg_corrected'] = weights['sigma'] / np.log(np.array(deg_map.a))
    #weights['pagerank_d100'] = np.array(pagerank(network, damping=1.).a)
    #weights['pagerank_d85'] = np.array(pagerank(network, damping=0.85).a)
    #weights['pagerank_d50'] = np.array(pagerank(network, damping=0.5).a)
    #weights['pagerank_d25'] = np.array(pagerank(network, damping=0.25).a)
    #weights['pagerank_d0'] = np.array(pagerank(network, damping=0.0).a) #equal adj. mat.
    #weights['katz'] = np.array(katz(network).a)
    #weights['cosine'] = calc_cosine(A)
    #weights['common_neighbours'] = calc_common_neigh(A)


    # filter out metrics containing nans or infs
    if False:
        # filter out metrics containing nans or infs
        weights = {key: val for key, val in weights.iteritems() if val is not None and np.isnan(val).sum() == 0 and np.isinf(val).sum() == 0}
    else:
        # replace nans and infs with zero
        for key, val in weights.iteritems():
            if val is not None:
                num_nans = np.isnan(val).sum()
                num_infs = np.isinf(val).sum()
                if num_nans > 0 or num_infs > 0:
                    print '[', name, '] ', key, ': shape:', val.shape, '|replace nans(', num_nans, ') and infs (', num_infs, ') of metric with zero'
                    val[np.isnan(val) | np.isinf(val)] = 0
                    weights[key] = val

    weights = {key.replace(' ', '_'): val for key, val in weights.iteritems()}

    entropy_df = pd.DataFrame()
    sort_df = []
    print '[', name, '] calc graph-layout'
    try:
        pos = sfdp_layout(network, groups=network.vp['com'], mu=3.0)
    except KeyError:
        pos = sfdp_layout(network)

    corr_df = pd.DataFrame(columns=['deg'], data=deg_map.a)
    stat_distributions = {}
    for key, weight in sorted(weights.iteritems(), key=operator.itemgetter(0)):
        print '[', name, '||', key, '] start calc'
        #print 'weight', weight
        ent, stat_dist = calc_entropy_and_stat_dist(A, weight)
        stat_distributions[key] = stat_dist
        print '[', name, '||', key, '] entropy rate:', ent
        entropy_df.at[0, key] = ent
        sort_df.append((key, ent))
        #print 'draw graph:', out_dir + name + '_' + key
        corr_df[key] = stat_dist
    base_line_abs_vals = stat_distributions['adjacency']
    #base_line = np.array([[1. / network.num_vertices()]])
    base_line = base_line_abs_vals / 100  # /100 for percent
    #vertex_size = deg_map
    vertex_size = network.new_vertex_property('float')
    vertex_size.a = base_line
    min_stat_dist = min([min(i) for i in stat_distributions.values()])
    max_stat_dist = max([max(i) for i in stat_distributions.values()])
    min_val = min([min(i/base_line) for i in stat_distributions.values()])
    max_val = max([max(i/base_line) for i in stat_distributions.values()])
    trapped_df = pd.DataFrame(index=range(network.num_vertices()))
    if True:
        all_vals = [j for i in stat_distributions.values() for j in i / base_line]
        max_val = np.mean(all_vals) + (2 * np.std(all_vals))

    for key, stat_dist in sorted(stat_distributions.iteritems(), key=operator.itemgetter(0)):
        weight = weights[key]
        stat_dist_diff = stat_dist / base_line
        stat_dist_diff[np.isclose(stat_dist_diff, 100.0)] = 100.0
        draw_graph(network, color=stat_dist_diff, min_color=min_val, max_color=max_val, sizep=deg_map,
                   groups='com', output=out_dir + name + '_graph_' + key, pos=pos)
        plt.close('all')

        stat_dist_ser = pd.Series(data=stat_dist)
        x = ('stationary value of adjacency', base_line_abs_vals)
        y = (key.replace('_', ' ') + ' difference', stat_dist_diff / 100)
        create_scatter(x=x, y=y, fname=out_dir + name + '_scatter_' + key, c=weight)

        # plot stationary distribution
        #stat_dist_ser.plot(kind='hist', bins=25, range=(min_stat_dist, max_stat_dist), lw=0, normed=True)
        #plt.title(key)
        #plt.ylabel('#nodes')
        #plt.xlabel('stationary value')
        #plt.savefig(out_dir + name + '_stat_dist_' + key + '.png', bbox_tight=True)
        #plt.close('all')
        stat_dist_ser.sort(ascending=False)
        stat_dist_ser.index = range(len(stat_dist_ser))
        trapped_df[key] = stat_dist_ser.cumsum()
        trapped_df[key] /= trapped_df[key].max()
    trapped_df['uniform'] = np.array([1]*len(trapped_df)).cumsum()
    trapped_df['uniform'] /= trapped_df['uniform'].max()
    trapped_df.index += 1
    trapped_df['idx'] = np.round(np.array(trapped_df.index).astype('float') / len(trapped_df) * 100)
    if len(trapped_df) > 20:
        trapped_df['idx'] = trapped_df['idx'].astype('int')
        trapped_df['idx'] = trapped_df['idx'].apply(lambda x: int(x / 10) * 10 if x > 10 else x)
        trapped_df.drop_duplicates(subset=['idx'], inplace=True)
        #trapped_df.drop('idx', axis=1, inplace=True)
    #trapped_df.index = np.round(np.array(trapped_df.index).astype('float') / len(trapped_df) * 100)

    trapped_df.plot(x='idx',lw=2, alpha=0.5, style=['-o', '-v', '-^', '-s', '-*', '-D'], logx=True)
    ticks = [1, 3, 5, 10, 30, 50, 100]
    #plt.xscale('log')
    plt.xticks(ticks, [i + '%' for i in map(str, ticks)])
    plt.xlim([0.9, 110])
    plt.yticks([0, .25, .5, .75, 1], ['0', '25%', '50%', '75%', '100%'])
    plt.xlabel('percent of nodes')
    plt.ylim([0, 1])
    plt.ylabel('cumulative sum of stationary distribution values')
    plt.savefig(out_dir + name + '_trapped.png', bbox_tight=True)
    plt.close('all')

    try:
        num_cols = len(corr_df.columns) * 3
        pd.scatter_matrix(corr_df, figsize=(num_cols, num_cols), diagonal='kde', range_padding=0.2, grid=True)
        plt.savefig(out_dir + name + '_scatter_matrix.png', bbox_tight=True)
        plt.close('all')
    except:
        pass
    #entropy_df.sort(axis=1, inplace=True)
    sorted_keys, sorted_values = zip(*sorted(sort_df, key=lambda x: x[1], reverse=True))
    if len(set(sorted_values)) == 1:
        sorted_keys = sorted(sorted_keys)
    entropy_df = entropy_df[list(sorted_keys)]
    print '[', name, '] entropy rates:'
    print entropy_df
    ax = entropy_df.plot(kind='bar', label=[i.replace('_', ' ') for i in entropy_df.columns])
    min_e, max_e = entropy_df.loc[0].min(), entropy_df.loc[0].max()
    ax.set_ylim([min_e * 0.99, max_e * 1.01])
    plt.ylabel('entropy rate')
    plt.legend(loc='upper left')
    plt.xlim([-1, 0.4])
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.savefig(out_dir + name + '_entropy_rates.png', bbox_tight=True)
    plt.close('all')

#=======================================================================================================================


def main():
    generator = SBMGenerator()
    granularity = 10
    num_samples = 10
    base_outdir = 'output/'
    basics.create_folder_structure(base_outdir)
    font_size = 12
    matplotlib.rcParams.update({'font.size': font_size})

    test = False
    multip = True
    first_two_only = True
    if first_two_only:
        multip = False
    worker_pool = multiprocessing.Pool(processes=14)
    if test:
        outdir = base_outdir + 'tests/'
        basics.create_folder_structure(outdir)

        print 'complete graph'.center(80, '=')
        name = 'complete_graph_n50'
        net = complete_graph(10)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)

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

        print 'quick tests done'.center(80, '=')
    else:
        num_links = 300
        num_nodes = 100
        num_blocks = 3
        print 'karate'.center(80, '=')
        name = 'karate'
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = load_edge_list('/opt/datasets/karate/karate.edgelist')
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        #print 'complete graph'.center(80, '=')
        #name = 'complete_graph_n' + str(num_nodes)
        #net = complete_graph(num_nodes)
        #generator.analyse_graph(net, outdir + name, draw_net=False)
        #if multip:
        #    worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
        #                            callback=None)
        #else:
        #    self_sim_entropy(net, name=name, out_dir=outdir)

        print 'sbm'.center(80, '=')
        name = 'sbm_strong_n' + str(num_nodes) + '_m' + str(num_links)
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.05)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        if first_two_only:
            exit()
        print 'sbm'.center(80, '=')
        name = 'sbm_weak_n' + str(num_nodes) + '_m' + str(num_links)
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=num_blocks, num_links=num_links, other_con=0.5)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)

        #print 'powerlaw'.center(80, '=')
        #name = 'powerlaw_n' + str(num_nodes) + '_m' + str(num_links)
        #net = generator.gen_stock_blockmodel(num_nodes=num_nodes, blocks=1, num_links=num_links)
        #generator.analyse_graph(net, outdir + name, draw_net=False)
        #if multip:
        #    worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
        #                            callback=None)
        #else:
        #    self_sim_entropy(net, name=name, out_dir=outdir)

        print 'price network'.center(80, '=')
        name = 'price_net_n' + str(num_nodes) + '_m' + str(net.num_edges())
        outdir = base_outdir + name + '/'
        basics.create_folder_structure(outdir)
        net = price_network(num_nodes, m=2, gamma=1, directed=False)
        generator.analyse_graph(net, outdir + name, draw_net=False)
        if multip:
            worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                    callback=None)
        else:
            self_sim_entropy(net, name=name, out_dir=outdir)
        if True:
            print 'wiki4schools'.center(80, '=')
            name = 'wiki4schools'
            outdir = base_outdir + name + '/'
            basics.create_folder_structure(outdir)
            net = load_edge_list('/opt/datasets/wikiforschools/graph')
            net.vp['com'] = load_property(net, '/opt/datasets/wikiforschools/artid_catid', type='int')
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)
            print 'facebook'.center(80, '=')
            name = 'facebook'
            outdir = base_outdir + name + '/'
            basics.create_folder_structure(outdir)
            net = load_edge_list('/opt/datasets/facebook/facebook')
            net.vp['com'] = load_property(net, '/opt/datasets/facebook/facebook_com', type='int', line_groups=True)
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)
            print 'youtube'.center(80, '=')
            name = 'youtube'
            outdir = base_outdir + name + '/'
            basics.create_folder_structure(outdir)
            net = load_edge_list('/opt/datasets/youtube/youtube')
            net.vp['com'] = load_property(net, '/opt/datasets/youtube/youtube_com', type='int', line_groups=True)
            if multip:
                worker_pool.apply_async(self_sim_entropy, args=(net,), kwds={'name': name, 'out_dir': outdir},
                                        callback=None)
            else:
                self_sim_entropy(net, name=name, out_dir=outdir)
            print 'dblp'.center(80, '=')
            name = 'dblp'
            outdir = base_outdir + name + '/'
            basics.create_folder_structure(outdir)
            net = load_edge_list('/opt/datasets/dblp/dblp')
            net.vp['com'] = load_property(net, '/opt/datasets/dblp/dblp_com', type='int', line_groups=True)
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