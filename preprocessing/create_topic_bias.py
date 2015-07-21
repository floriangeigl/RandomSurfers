from __future__ import division
import random
from collections import defaultdict, Counter
import operator
from tools.gt_tools import load_edge_list
from graph_tool.all import *
import numpy as np
from data_io import try_dump

net_name = '/opt/datasets/orf_tvthek/tvthek_orf'
topics = ['Nationalrat', 'Steiermark-heute', 'Raetselburg']
names = ['topic_one', 'topic_two', 'topic_three']
bias_factor = 2.
g = load_edge_list(net_name, vertex_id_dtype='string', directed=True)
lcc = label_largest_component(g, directed=True)
g.set_vertex_filter(lcc)
g.purge_vertices()
g.clear_filters()
cat = g.vp['category']
cat_dist = sorted(Counter([cat[v] for v in g.vertices()]).iteritems(), key=lambda x: x[1], reverse=True)
print '\n'.join(
    map(str, cat_dist[:20]))
print '...'
print '\n'.join(
    map(str, cat_dist[-20:]))
for t, n in zip(topics, names):
    bias = np.array([bias_factor if cat[v] == t else 1. for v in g.vertices()])
    dump_name = net_name + '_' + n + '.bias'
    print 'save topic bias:', t, '->', dump_name
    try_dump(bias, dump_name)
