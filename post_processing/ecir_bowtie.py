from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from graph_tool.all import *
from collections import defaultdict
import random
import network_matrix_tools
import pandas as pd
from tools.mpl_tools import plot_set_limits
import datetime


def bias_pr(node_ids, adjacency_matrix, bias_strength=2):
    bias_vec = np.ones(adjacency_matrix.shape[0])
    bias_vec[node_ids] = bias_strength
    _, biased_stat_dist = network_matrix_tools.calc_entropy_and_stat_dist(adjacency_matrix, bias_vec, method='PR',
                                                                          smooth_bias=False, calc_entropy_rate=False,
                                                                          verbose=False)
    return biased_stat_dist[node_ids].sum()

def main():
    base_dir = '/home/fgeigl/navigability_of_networks/output/ecir_synthetic_coms/'
    g = load_graph(base_dir + 'bow_tie.gt')
    out_dir = base_dir
    print g
    print g.vp.keys()
    bow_tie_pmap = g.vp['bowtie']

    nodes_mapping = defaultdict(set)
    for v in g.vertices():
        nodes_mapping[bow_tie_pmap[v]].add(int(v))
    print {key: len(val) for key, val in nodes_mapping.iteritems()}
    print {key: len(val)/g.num_vertices() for key, val in nodes_mapping.iteritems()}
    sample_size = [0.05]
    biased_comps = ['SCC', 'IN', 'OUT']
    samples = 10
    bias_strength = [1, 2, 5, 10, 25, 50, 75, 100, 200]

    _, orig_pr = network_matrix_tools.calc_entropy_and_stat_dist(adjacency(g), method='PR',
                                                                                  smooth_bias=False,
                                                                                  calc_entropy_rate=False,
                                                                                  verbose=False)

    adjacency_matrix = adjacency(g)
    for s_size in sample_size:
        s_size_nodes = int(round(s_size * g.num_vertices()))
        groups = list()
        for c_name in biased_comps:
            for s_idx in range(samples):
                groups.append((c_name, random.sample(nodes_mapping[c_name], s_size_nodes)))
        df = pd.DataFrame(columns=['component', 'node-ids'], data=groups)
        # df['num_nodes'] = df['node-ids'].apply(len)
        df['orig_pr'] = df['node-ids'].map(lambda x: orig_pr[x].sum())
        for bs in bias_strength:
            df['biased_pr_' + str(bs)] = df['node-ids'].apply(bias_pr, args=(adjacency_matrix, bs))
        df.drop('node-ids', inplace=True, axis=1)
        grp_df = df.groupby('component').mean()
        plot_df = grp_df.T.copy()
        plot_df.drop(filter(lambda x: 'biased_pr' not in x, plot_df.index), inplace=True)
        plot_df['bs'] = plot_df.index
        plot_df['bs'] = plot_df['bs'].map(lambda x: int(x.rsplit('_', 1)[-1]))
        plot_df.plot(x='bs', lw=3)
        plt.xlabel('bias strength')
        plt.ylabel(r'$\pi_g^b$')
        plt.xlim([0, max(bias_strength)])
        plt.savefig(out_dir + 'bow_tie_iter_bs.pdf')
        plt.close('all')

        grp_df.plot.bar(x=grp_df.index, y='orig_pr', legend=False, lw=0, alpha=0.8)
        plt.ylabel(r'$\pi_g^b$')
        plot_set_limits(axis='y', values=grp_df['orig_pr'])
        plt.savefig(out_dir + 'bow_tie_pr.pdf')
        plt.close('all')

        grp_df['bias_fac'] = grp_df['biased_pr_2'] / grp_df['orig_pr']
        grp_df.plot.bar(x=grp_df.index, y='bias_fac', legend=False, lw=0, alpha=0.8)
        plt.ylabel(r'$\pi_g^b$')
        plot_set_limits(axis='y', values=grp_df['bias_fac'])
        plt.savefig(out_dir + 'bow_tie_pr_fac.pdf')
        plt.close('all')

if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start
