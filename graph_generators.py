from __future__ import division
from sys import platform as _platform
import matplotlib
from collections import defaultdict
import operator

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg', warn=False)
import matplotlib.pylab as plt
from graph_tool.all import *
import pandas as pd
import scipy.stats as stats
import numpy as np


def gen_stock_blockmodel(num_nodes=100, blocks=3, self_con=1, other_con=0.2, directed=False, degree_corrected=True,
                         powerlaw_exp=2.1, num_links=300, loops=False):
    g = Graph(directed=directed)
    com_pmap = g.new_vertex_property('int')
    for idx in range(num_nodes):
        com_pmap[g.add_vertex()] = idx % blocks
    g.vp['com'] = com_pmap
    other_con /= (blocks - 1)

    prob_pmap = g.new_vertex_property('float')
    block_to_vertices = dict()
    block_to_cumsum = dict()
    for i in range(blocks):
        vertices_in_block = list(filter(lambda x: com_pmap[x] == i, g.vertices()))
        block_to_vertices[i] = vertices_in_block
        if degree_corrected:
            powerlaw_dist = 1 - stats.powerlaw.rvs(powerlaw_exp, size=len(vertices_in_block))
        else:
            powerlaw_dist = np.random.random(size=len(vertices_in_block))
        powerlaw_dist /= powerlaw_dist.sum()
        cum_sum = np.cumsum(powerlaw_dist)
        assert np.allclose(cum_sum[-1], 1)
        block_to_cumsum[i] = cum_sum
        for v, p in zip(vertices_in_block, powerlaw_dist):
            prob_pmap[v] = p
    blocks_prob = list()
    for i in range(blocks):
        row = list()
        for j in range(blocks):
            if i == j:
                row.append(self_con)
            else:
                row.append(other_con)
        blocks_prob.append(np.array(row))
    blocks_prob = np.array(blocks_prob)
    blocks_prob /= blocks_prob.sum()
    # print blocks_prob
    cum_sum = np.cumsum(blocks_prob)
    assert np.allclose(cum_sum[-1], 1)
    # print cum_sum
    links_created = 0
    for v in g.vertices():
        if True or v.in_degree() + v.out_degree() == 0:
            src_block = com_pmap[v]
            while True:
                dest_b = get_one_random_block(cum_sum, blocks, src_block)
                dest_v = block_to_vertices[dest_b][get_random_node(block_to_cumsum[dest_b])]
                if loops or v != dest_v and g.edge(v, dest_v) is None:
                    g.add_edge(v, dest_v)
                    links_created += 1
                    break

    for link_idx in range(num_links - links_created):
        edge = 1
        src_v, dest_v = None, None
        while edge is not None:
            src_b, dest_b = get_random_blocks(cum_sum, blocks)
            src_v = block_to_vertices[src_b][get_random_node(block_to_cumsum[src_b])]
            dest_v = block_to_vertices[dest_b][get_random_node(block_to_cumsum[dest_b])]
            edge = g.edge(src_v, dest_v)
            if edge is None and not loops and src_v == dest_v:
                edge = 1
        g.add_edge(src_v, dest_v)
    return g


def get_random_node(cum_sum):
    rand_num = np.random.random()
    idx = 0
    for idx, i in enumerate(cum_sum):
        if i >= rand_num:
            return idx
    print 'warn: get rand num till end'
    return idx


def get_random_blocks(cum_sum, num_blocks):
    rand_num = np.random.random()
    idx = 0
    for idx, i in enumerate(cum_sum):
        if i >= rand_num:
            return idx % num_blocks, int(idx / num_blocks)
    print 'warn: get rand block till end'
    return idx % num_blocks, int(idx / num_blocks)


def get_one_random_block(cum_sum, num_blocks, row):
    src_b = None
    dest_b = None
    while src_b is None or row != src_b:
        src_b, dest_b = get_random_blocks(cum_sum, num_blocks)
    return dest_b


def analyse_graph(g):
    print str(g)
    deg_map = g.degree_property_map('total')
    plt.close('all')
    ser = pd.Series(deg_map.a)
    ser.plot(kind='hist', bins=20, lw=0)
    plt.xlabel('degree')
    plt.ylabel('num nodes')
    plt.show()
    plt.close('all')
    graph_draw(g, vertex_fill_color=g.vp['com'], vertex_size=prop_to_size(deg_map, mi=2, ma=15))
    plt.show()
    print '=' * 80