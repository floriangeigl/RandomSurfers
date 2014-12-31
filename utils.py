from __future__ import division
from sys import platform as _platform
from graph_tool.all import *
from scipy.stats import poisson, powerlaw
import operator
import pandas as pd
import numpy as np
import matplotlib


if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt


def graph_gen(self_con, other_con, nodes=100, groups=10):
    corr = lambda x, y: self_con if x == y else other_con
    g, bm = random_graph(nodes, lambda: (powerlaw.rvs(2.4, size=1) * -1 + 1) * 20, directed=False, model="blockmodel-traditional", block_membership=lambda: np.random.randint(int(groups)), vertex_corr=corr)
    # poisson(10).rvs(1)
    return g, bm


def plot_measurements_of_ranking(ranking, measurements_dict, filename=None, logx=True, logy=True):
    df = pd.DataFrame(columns=['ranking'], data=ranking)
    for key, pmap in measurements_dict.iteritems():
        net = pmap.get_graph()
        df[key] = df['ranking'].apply(lambda x: pmap[net.vertex(x)])
    df.drop('ranking', axis=1, inplace=True)
    df.plot(lw=2, alpha=0.8, logx=logx, logy=logy)
    plt.ylabel('ranking value')
    plt.xlabel('ranked vertices')
    outfname = 'output/measurements_of_ranking.png' if filename is None else filename
    plt.savefig(outfname, dpi=150)
    plt.close('all')


def get_ranking(vp_map):
    network = vp_map.get_graph()
    result = map(int, sorted(network.vertices(), key=lambda v: vp_map[v], reverse=True))
    return result


def get_ranking_df(ranking, weights):
    data = [(val, vertex) for val, vertex in zip(weights, ranking)]
    df = pd.DataFrame(columns=['values', 'ranked_vertex'], data=data)
    return df
