from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
from tools.gt_tools import GraphGenerator
import cost_function
import moves
import optimizer
import pandas as pd
from tools.basics import *
from graph_tool.all import *
import seaborn as sns
import numpy as np
import random
import operator
import os
from utils import *
import datetime
from collections import defaultdict


def main():
    # total number of nodes per network
    nodes = 1000

    # divide network into x groups
    groups = 3

    # stepsize between different block configurations
    step = 0.2

    # max connection between blocks
    max_con = 1

    # reduce targets by
    target_reduce = 0.1

    # max runs for optimizer
    max_runs = 1000

    # percentage of which to calc the correlation between rankings (top x)
    correlation_perc = 0.2

    # the weighting function for the ranking
    ranking_weights_func = lambda x: np.power(max(x - (nodes * 0.8), 0), 2)

    results_df = pd.DataFrame()
    results_df.index = results_df.index.astype('float')
    assert groups > 0
    for network_num, self_con in enumerate(np.arange(0, max_con + (step * 0.9), step)):

        # generate network
        self_con = max_con - self_con
        other_con = (max_con - self_con) / (groups - 1)
        print 'gen graph with ', nodes, 'nodes.(self:', self_con, ',other:', other_con, ')'
        network, bm_groups = graph_gen(self_con, other_con, nodes, groups)

        # init cost function and print configuration details
        cf = cost_function.CostFunction(network, target_reduce=target_reduce, ranking_weights=[ranking_weights_func(i) for i in reversed(range(network.num_vertices()))], verbose=0)
        print 'missions:', sum(len(j) for i, j in cf.pairs), 'targets:', len(cf.pairs)
        if network_num == 0:
            plt.clf()
            plt.plot(cf.ranking_weights, lw=4, label='ranking weights')
            plt.savefig('output/sbm_results_rank_weights.png')
            plt.close('all')

        mover = moves.MoveTravelSM(verbose=0)
        all_nodes = range(network.num_vertices())
        random.shuffle(all_nodes)

        # optimize ranking
        print 'optimizing...'
        opt = optimizer.SimulatedAnnealing(cf, mover, all_nodes, known=0.1, max_runs=max_runs, reduce_step_after_fails=0, reduce_step_after_accepts=100, verbose=0)
        ranking, cost = opt.optimize()

        # create ranking dataframe
        df = get_ranking_df(ranking, cf.ranking_weights)

        # print some stats
        print 'runs:', opt.runs
        print 'cost:', cost
        print df.head()
        print 'ranking top 50', ranking[:50]
        targets, _ = zip(*cf.pairs)
        sample_size = min(len(targets), 50)
        print sample_size, 'targets', random.sample(targets, sample_size)
        targets = set(targets)
        print 'top 50 in targets:', np.sum(i in targets for i in ranking[:50])

        # get ranking of different measurements
        deg_pmap = network.degree_property_map('total')
        df['deg'] = get_ranking(deg_pmap)
        test_dict = {network.vertex(v): idx for idx, v in enumerate(reversed(ranking))}
        vp_map = network.new_vertex_property('int')
        for v in network.vertices():
            vp_map[v] = test_dict[v]
        df['test'] = get_ranking(vp_map)
        if network.is_directed():
            vp_map = network.degree_property_map('in')
            df['in-deg'] = get_ranking(vp_map)
            vp_map = network.degree_property_map('out')
            df['out-deg'] = get_ranking(vp_map)
        pr_pmap = pagerank(network)
        df['pagerank'] = get_ranking(pr_pmap)
        bw_pmap, _ = betweenness(network)
        df['betweeness'] = get_ranking(bw_pmap)

        # plot networks
        create_folder_structure('output/graph_plots/')
        eweight = network.new_edge_property('float')
        for e in network.edges():
            eweight[e] = 1 if bm_groups[e.source()] == bm_groups[e.target()] else 0
        pos = sfdp_layout(network, groups=bm_groups, mu=1)
        graph_draw(network, pos=pos, vertex_fill_color=bm_groups, output='output/graph_plots/sbm_' + str(self_con) + '.png')
        graph_draw(network, pos=pos, vertex_fill_color=deg_pmap, output='output/graph_plots/sbm_' + str(self_con) + '_deg.png')
        graph_draw(network, pos=pos, vertex_fill_color=bw_pmap, output='output/graph_plots/sbm_' + str(self_con) + '_betwe.png')

        # plot degree distribution
        deg_dist = defaultdict(float)
        for v in network.vertices():
            deg_dist[deg_pmap[v]] += 1
        x_axis = range(max(deg_dist.keys()) + 1)
        plt.plot(x=x_axis, y=[deg_dist[i] for i in x_axis], lw=2)
        plt.title('degree distribution')
        plt.xscale('log')
        plt.yscale('log')
        plt.xtitle('degree')
        plt.ytitle('# nodes')
        plt.savefig('output/graph_plots/sbm_' + str(self_con) + '_degdist.png')
        plt.close('all')

        # calculate correlation, append it to overall results and plot it
        if correlation_perc > 0:
            print 'calc correlation between top:', correlation_perc
            df = df.loc[0:int(round(len(df) * correlation_perc))]
        correlations = df.corr(method='spearman')
        print correlations
        results_df.at[self_con, 'deg'] = correlations['deg']['ranked_vertex']
        results_df.at[self_con, 'pagerank'] = correlations['pagerank']['ranked_vertex']
        results_df.at[self_con, 'betweeness'] = correlations['betweeness']['ranked_vertex']
        results_df.plot(lw=3)
        plt.xlabel('self connectivity')
        plt.savefig('output/sbm_results.png', dpi=300)
        plt.close('all')


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print 'ALL DONE. Time:', datetime.datetime.now() - start

