from graph_tool.all import *
from scipy.stats import poisson
import operator
import pandas as pd
import numpy as np


def graph_gen(self_con, other_con, nodes=100, groups=10):
    corr = lambda x, y: self_con if x == y else other_con
    g, bm = random_graph(nodes, lambda: poisson(10).rvs(1), directed=False, model="blockmodel-traditional", block_membership=lambda: np.random.randint(int(groups)), vertex_corr=corr)
    return g, bm


def get_ranking(vp_map):
    network = vp_map.get_graph()
    result = map(int, sorted(network.vertices(), key=lambda v: vp_map[v], reverse=True))
    return result


def get_ranking_df(ranking, weights):
    data = [(val, vertex) for val, vertex in zip(weights, ranking)]
    df = pd.DataFrame(columns=['values', 'ranked_vertex'], data=data)
    return df
