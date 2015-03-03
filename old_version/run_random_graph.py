from __future__ import division
from sys import platform as _platform

import matplotlib


if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
from tools.gt_tools import GraphGenerator
from old_version import optimizer, moves, cost_function
import random
import pandas as pd
import numpy as np

network = GraphGenerator(200)
network = network.create_random_graph()
cf = cost_function.CostFunction(network, pairs_reduce=0.3, ranking_weights=[np.power(i, 2) for i in reversed(range(network.num_vertices()))])
mover = moves.MoveTravelSM()
all_nodes = range(network.num_vertices())
random.shuffle(all_nodes)
opt = optimizer.SimulatedAnnealing(cf, mover, all_nodes, known=0.1, max_runs=1000, reduce_step_after_fails=0, reduce_step_after_accepts=100)
ranking, cost = opt.optimize()
print 'runs:', opt.runs
print 'best ranking', ranking
print 'cost:', cost
print 'weights:', cf.ranking_weights
data = [(idx, val) for idx, val in enumerate(cf.ranking_weights)]
df = pd.DataFrame(columns=['ranking', 'values'], data=data, index=ranking)
df.sort(inplace=True)
# deg
vp_map = network.degree_property_map('total')
df['deg'] = [vp_map[v] for v in network.vertices()]

if network.is_directed():
    vp_map = network.degree_property_map('in')
    df['in-deg'] = [vp_map[v] for v in network.vertices()]

    vp_map = network.degree_property_map('out')
    df['out-deg'] = [vp_map[v] for v in network.vertices()]

vp_map = pagerank(network)
df['pagerank'] = [vp_map[v] for v in network.vertices()]

vp_map, _ = betweenness(network)
df['betweeness'] = [vp_map[v] for v in network.vertices()]

print df.head()
pd.scatter_matrix(df, alpha=0.2)
plt.savefig('random_graph.png', dpi=300)
plt.close('all')
plt.plot(x=range(len(cf.ranking_weights)), y=cf.ranking_weights, label='ranking weights')
plt.savefig('random_graph_rw.png')


