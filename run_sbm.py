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
from scipy.stats import poisson
import random
import operator
import os


def graph_gen(self_con, other_con, nodes=100, groups=10):
    corr = lambda x, y: self_con if x == y else other_con
    g, bm = random_graph(nodes, lambda: poisson(10).rvs(1), directed=False, model="blockmodel-traditional", block_membership=lambda: np.random.randint(int(groups)), vertex_corr=corr)
    return g, bm


def get_ranking(vp_map):
    network = vp_map.get_graph()
    tmp_ranking = [(int(v), rank) for rank, v in zip(xrange(network.num_vertices()), sorted(network.vertices(), key=lambda x: vp_map[x], reverse=True))]
    _, result = zip(*sorted(tmp_ranking, key=operator.itemgetter(0)))
    return result


def main():
    nodes = 100
    groups = 3
    step = 0.01
    max_con = 1
    pairs_reduce = 0.1
    max_runs = 10000
    results_df = pd.DataFrame()
    results_df.index = results_df.index.astype('float')
    correlation_perc = -1
    for self_con in np.arange(0, max_con + (step * 0.9), step):
        self_con = max_con - self_con
        print 'gen graph with ', nodes, 'nodes.(self:', self_con, ',other:', max_con - self_con, ')'
        network, bm_groups = graph_gen(self_con, max_con - self_con, nodes, groups)
        cf = cost_function.CostFunction(network, pairs_reduce=pairs_reduce, ranking_weights=[np.power(i, 2) for i in reversed(range(network.num_vertices()))], verbose=0)
        mover = moves.MoveTravelSM(verbose=0)
        all_nodes = range(network.num_vertices())
        random.shuffle(all_nodes)
        print 'optimizing...'
        opt = optimizer.SimulatedAnnealing(cf, mover, all_nodes, known=0.1, max_runs=max_runs, reduce_step_after_fails=0, reduce_step_after_accepts=100, verbose=0)
        ranking, cost = opt.optimize()
        print 'runs:', opt.runs
        # print 'best ranking', ranking
        print 'cost:', cost
        # print 'weights:', cf.ranking_weights
        data = [(idx, val) for idx, val in enumerate(cf.ranking_weights)]
        df = pd.DataFrame(columns=['ranking', 'values'], data=data, index=ranking)
        df.sort(inplace=True)
        # deg
        vp_map = network.degree_property_map('total')
        df['deg'] = get_ranking(vp_map)

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
        graph_draw(network, groups=bm_groups, output='output/graph_plots/sbm_' + str(self_con) + '.png')
        plt.close('all')
        print correlations
        results_df.at[self_con, 'deg'] = correlations['deg']['ranking']
        results_df.at[self_con, 'pagerank'] = correlations['pagerank']['ranking']
        results_df.at[self_con, 'betweeness'] = correlations['betweeness']['ranking']
        results_df.plot(lw=3)
        plt.xlabel('self connectivity')
        plt.savefig('output/sbm_results.png', dpi=300)
        plt.close('all')

    plt.plot(x=range(len(cf.ranking_weights)), y=cf.ranking_weights, label='ranking weights')
    plt.savefig('output/sbm_results_rank_weights.png')


if __name__ == '__main__':
    main()

