from graph_tool.all import *
from scipy.stats import poisson
import operator
import numpy as np


def graph_gen(self_con, other_con, nodes=100, groups=10):
    corr = lambda x, y: self_con if x == y else other_con
    g, bm = random_graph(nodes, lambda: poisson(10).rvs(1), directed=False, model="blockmodel-traditional", block_membership=lambda: np.random.randint(int(groups)), vertex_corr=corr)
    return g, bm


def get_ranking(vp_map):
    network = vp_map.get_graph()
    tmp_ranking = ((rank, int(v)) for rank, v in zip(xrange(network.num_vertices()), sorted(network.vertices(), key=lambda x: vp_map[x], reverse=True)))
    _, result = zip(*sorted(tmp_ranking, key=operator.itemgetter(0)))
    return result
