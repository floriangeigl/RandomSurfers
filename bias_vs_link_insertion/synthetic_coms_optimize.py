from __future__ import division, print_function

from sys import platform as _platform

import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
from matplotlib import gridspec
from post_processing.plotting import *
import os
from tools.basics import create_folder_structure, find_files
import traceback
import numpy as np
from graph_tool.all import *
import datetime
import network_matrix_tools
import random
import tools.mpl_tools as plt_tools
from scipy.sparse import diags, csr_matrix
import sys
import itertools
from collections import Counter

pd.set_option('display.width', 600)
pd.set_option('display.max_colwidth', 600)
matplotlib.rcParams.update({'font.size': 25})
default_x_ticks_pad = matplotlib.rcParams['xtick.major.pad']
default_y_ticks_pad = matplotlib.rcParams['xtick.major.pad']
matplotlib.rcParams['xtick.major.pad'] *= 2
matplotlib.rcParams['ytick.major.pad'] *= 2


class GetOutOfLoops(Exception):
    pass


def mix_bias_linkins_and_calc((sample_size, com_nodes), bias_strength=2, mixture=0.5, net=None, num_links=1,
                              top_measure=None, verbose=False):
    if sample_size > 0.21:
        # print 'skip sample-size:', sample_size
        return np.nan
    if not np.isclose(sample_size, mix_bias_linkins_and_calc.sample_size):
        mix_bias_linkins_and_calc.sample_size = sample_size
        mix_bias_linkins_and_calc.calc_counter = 0
        mix_bias_linkins_and_calc.ss_string = "%.3f" % sample_size
        print('')
    orig_adj = adjacency(net)
    if isinstance(num_links, str):
        if num_links == 'fair':
            bias_m = np.zeros(net.num_vertices()).astype('int')
            bias_m[com_nodes] = 1
            bias_m = diags(bias_m, 0)
            num_links = int(bias_m.dot(adjacency(net)).sum()) * (bias_strength - 1)
            # print 'fair links:', num_links
    print_num_links = (str(num_links / 1000) + 'k') if num_links > 1000 else num_links

    bias_links = int(np.round(num_links * mixture))
    top_new_links = num_links - bias_links
    if verbose:
        print('')
        print('mixture:', mixture)
        print('bias links:', bias_links)
        print('new links:', top_new_links)
        print('--')
        print('orig')
        print(orig_adj.todense())

    new_edges = list()
    orig_num_edges = net.num_edges()
    orig_num_com_nodes = len(com_nodes)
    if orig_num_com_nodes >= net.num_vertices():
        return None
    other_nodes = set(range(0, net.num_vertices())) - set(com_nodes)

    if top_new_links > 0:
        # create top-measure links
        if top_measure is None:
            nodes_measure = np.array(net.degree_property_map('in').a)
        else:
            nodes_measure = top_measure

        sorted_other_nodes = sorted(other_nodes, key=lambda x: nodes_measure[x], reverse=True)
        sorted_com_nodes = sorted(com_nodes, key=lambda x: nodes_measure[x], reverse=True)
        new_edges = list()
        while True:
            block_size = int(np.sqrt(top_new_links-len(new_edges))) + 1
            all_com_nodes = False
            if block_size > len(com_nodes):
                all_com_nodes = True
                block_size = int((top_new_links-len(new_edges))/len(com_nodes)) + 1

            if all_com_nodes:
                sorted_com_nodes_block = sorted_com_nodes
            else:
                sorted_com_nodes_block = sorted_com_nodes[:block_size]
            sorted_other_nodes_block = sorted_other_nodes[:block_size]
            # if verbose:
            # print('\n')
            # print('all com nodes:', all_com_nodes)
            # print('needed links:', top_new_links - len(new_edges))
            # print('max link for block-size:', len(sorted_other_nodes_block) * len(sorted_com_nodes_block))
            # print('block-size:', block_size)
            # sorted_com_nodes_block = sorted_com_nodes
            new_edges.extend(list(
                    itertools.islice(
                            ((src, dest) for dest in sorted_com_nodes_block for src in sorted_other_nodes_block),
                            (top_new_links - len(new_edges)))))
            if len(new_edges) >= top_new_links:
                break
            elif verbose:
                print('could not insert all links:', len(new_edges), 'of', top_new_links, 'add parallel')
        assert len(new_edges) == top_new_links
        edge_counter = Counter(new_edges)
        indizes, num_e = map(list, zip(*edge_counter.iteritems()))
        rows, cols = map(list, zip(*indizes))
        top_edge_matrix = csr_matrix((list(num_e), (rows, cols)), shape=orig_adj.shape).T
        if verbose:
            print('new links')
            print(top_edge_matrix.todense())
    else:
        top_edge_matrix = csr_matrix(orig_adj.shape, dtype='float')

    # insert biased links
    if bias_links > 0:
        deg_map = net.degree_property_map('total')
        e_p_f = lambda e: (deg_map[e.source()] + deg_map[e.target()]) / 2.
        e_w = net.new_edge_property('int')
        e_w.a = map(e_p_f, net.edges())
        prob_mat = adjacency(net, weight=e_w).astype('float')
        prob_mat /= prob_mat.sum()
        prob_mat_idx = prob_mat.nonzero()
        prob_mat_prob = prob_mat.data
        bias_edges_idx = np.random.choice(range(len(prob_mat_prob)), size=min(bias_links, len(prob_mat_prob)),
                                          replace=False,
                                          p=prob_mat_prob)
        row_idx, col_idx = prob_mat_idx[0][bias_edges_idx], prob_mat_idx[1][bias_edges_idx]
        bias_cum_sum_links = (np.array(orig_adj[row_idx, col_idx]).flatten() * (bias_strength - 1)).cumsum()
        last_idx = np.searchsorted(bias_cum_sum_links, bias_links, side='right')
        if last_idx > 0:
            last_idx = min(last_idx, len(bias_cum_sum_links) - 1)
            last_idx = last_idx if abs(bias_links - bias_cum_sum_links[last_idx]) < abs(
                    bias_links - bias_cum_sum_links[last_idx - 1]) else (last_idx - 1)
        row_idx, col_idx = row_idx[:last_idx], col_idx[:last_idx]
        if verbose:
            print('biased links:', row_idx, col_idx)
        bias_mat = orig_adj.astype('bool').astype('float') + csr_matrix(
                ([bias_strength - 1] * len(row_idx), (row_idx, col_idx)),
                shape=orig_adj.shape)
        if verbose:
            print('bias')
            print(bias_mat.todense())

        biased_mat = orig_adj.multiply(bias_mat)
        if verbose:
            print('biased')
            print(biased_mat.todense())
    else:
        biased_mat = orig_adj

    combined_mat = biased_mat + top_edge_matrix

    if verbose:
        print('combined')
        print(combined_mat.todense())

    _, relinked_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(combined_mat, method='EV',
                                                                            smooth_bias=False,
                                                                            calc_entropy_rate=False, verbose=False)
    assert orig_num_com_nodes == len(com_nodes)
    relinked_stat_dist_sum = relinked_stat_dist[com_nodes].sum()
    assert net.num_edges() == orig_num_edges
    print('\r', mix_bias_linkins_and_calc.ss_string, mix_bias_linkins_and_calc.calc_counter, '#links:', print_num_links,
          ' || mod. #links:', int(combined_mat.sum()), '|| bias strength:', bias_strength, '|| mixture:', mixture,
          '|| stat sum:', relinked_stat_dist_sum, end='')
    sys.stdout.flush()
    return relinked_stat_dist_sum


def main():
    if False:
        net = price_network(100, m=4, directed=False)
        sample_size = 0.2
        com_nodes = np.random.choice(np.array(range(net.num_vertices())), size=int(sample_size * net.num_vertices()),
                                     replace=False)
        print(net)
        print('com nodes:', com_nodes)
        mix_bias_linkins_and_calc.sample_size = 0.
        com_stat_dist = list()
        for i in np.linspace(0, 1, num=11):
            com_stat_dist.append(
                    (i, mix_bias_linkins_and_calc((sample_size, com_nodes), mixture=i, net=net, bias_strength=2,
                                                  num_links='fair')))
        df = pd.DataFrame(data=com_stat_dist)
        print(df)
    else:

        base_dir = '/home/fgeigl/navigability_of_networks/output/bias_link_ins/'
        out_dir = base_dir + 'plots/'
        create_folder_structure(out_dir)
        mixture_range = np.linspace(0, 1, num=11)
        bias_strength_range = [2, 3, 5, 10, 20, 50, 100, 200]
        result_files = filter(lambda x: '_bs' in x, find_files(base_dir, '.df'))
        print(result_files)
        cors = list()
        all_dfs = list()
        net_name = ''
        net = None
        skipped_ds = set()
        skipped_ds.add('daserste')
        # skipped_ds.add('wiki4schools')
        skipped_ds.add('tvthek_orf')
        for i in sorted(filter(lambda x: 'preprocessed' not in x, result_files),
                        key=lambda x: (x, int(x.split('_bs')[-1].split('.')[0]))):
            current_net_name = i.rsplit('_bs', 1)[0]
            bias_strength = int(i.split('_bs')[-1].split('.')[0])
            if bias_strength > 2:
                print('skip bs:', bias_strength)
                continue
            elif any((i in current_net_name for i in skipped_ds)):
                print('skip ds:', current_net_name)
                continue
            if current_net_name != net_name:
                print('*' * 120)
                print('load network:', current_net_name.rsplit('/', 1)[-1])
                net = load_graph(current_net_name)
                net_name = current_net_name
            assert net is not None
            preprocessed_filename = i.rsplit('.df', 1)[0] + '_preprocessed.df'
            df = pd.read_pickle(preprocessed_filename)
            result = list()
            mix_bias_linkins_and_calc.sample_size = 0.
            df = df[df['sample-size'] < 2.1]
            df = df.reindex(np.random.permutation(df.index))
            for idx, data in df[['sample-size', 'node-ids']].iterrows():
                ss, node_ids = list(data)
                for bs in bias_strength_range:
                    for mix in mixture_range:
                        stat_dist = mix_bias_linkins_and_calc((ss, node_ids), mixture=mix, net=net, bias_strength=bs,
                                                              num_links='fair')
                        result.append((ss, mix, bs, stat_dist))
                mix_bias_linkins_and_calc.calc_counter += 1
                result_df = pd.DataFrame(data=result,
                                         columns=['sample-size', 'mixture', 'bias_strength', 'sample_stat_dist'])
                result_df.to_pickle(os.path.dirname(preprocessed_filename) + '/' + current_net_name.rsplit('/', 1)[
                    -1] + '_mixture_res.df')


if __name__ == '__main__':
    start = datetime.datetime.now()
    print('START:', start)
    main()
    print('ALL DONE. Time:', datetime.datetime.now() - start)

