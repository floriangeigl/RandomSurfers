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


def main():
    nodes = 500
    groups = 5
    step = 0.2
    max_con = 1
    target_reduce = 0.01
    max_runs = 10000
    correlation_perc = 0.2
    results_df = pd.DataFrame()
    results_df.index = results_df.index.astype('float')
    for self_con in np.arange(0, max_con + (step * 0.9), step):
        self_con = max_con - self_con
        print 'gen graph with ', nodes, 'nodes.(self:', self_con, ',other:', max_con - self_con, ')'
        network, bm_groups = graph_gen(self_con, max_con - self_con, nodes, groups)
        cf = cost_function.CostFunction(network, target_reduce=target_reduce, ranking_weights=[i for i in reversed(range(network.num_vertices()))], verbose=0)
        mover = moves.MoveTravelSM(verbose=0)
        all_nodes = range(network.num_vertices())
        random.shuffle(all_nodes)
        print 'optimizing...'
        opt = optimizer.SimulatedAnnealing(cf, mover, all_nodes, known=0.1, max_runs=max_runs, reduce_step_after_fails=0, reduce_step_after_accepts=100, verbose=0)
        ranking, cost = opt.optimize()
        print 'runs:', opt.runs
        print 'cost:', cost
        df = get_ranking_df(ranking, cf.ranking_weights)
        print df.head()
        # deg
        vp_map = network.degree_property_map('total')
        df['deg'] = get_ranking(vp_map)

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

        vp_map = pagerank(network)
        df['pagerank'] = get_ranking(vp_map)

        vp_map, _ = betweenness(network)
        df['betweeness'] = get_ranking(vp_map)

        if correlation_perc > 0:
            print 'calc correlation between top:', correlation_perc
            df = df.loc[0:int(round(len(df) * correlation_perc))]
        correlations = df.corr(method='spearman')
        create_folder_structure('output/graph_plots/')
        graph_draw(network, groups=bm_groups, mu=3, vertex_fill_color=bm_groups, output='output/graph_plots/sbm_' + str(self_con) + '.png')
        plt.close('all')
        print correlations
        results_df.at[self_con, 'deg'] = correlations['deg']['ranked_vertex']
        results_df.at[self_con, 'pagerank'] = correlations['pagerank']['ranked_vertex']
        results_df.at[self_con, 'betweeness'] = correlations['betweeness']['ranked_vertex']
        results_df.plot(lw=3)
        plt.xlabel('self connectivity')
        plt.savefig('output/sbm_results.png', dpi=300)
        plt.close('all')
    plt.clf()
    plt.plot(cf.ranking_weights, lw=4, label='ranking weights')
    plt.savefig('output/sbm_results_rank_weights.png')
    plt.close('all')


if __name__ == '__main__':
    main()

