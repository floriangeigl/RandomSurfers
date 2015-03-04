from __future__ import division
import graph_tool as gt
import random
import operator
from collections import defaultdict
import numpy as np
import multiprocessing
import os
import sys
import utils as ut

def random_walk(network, max_steps, avoid_revisits=True, num_pairs=1000):
    c_list = ut.get_colors_list()
    p_id = multiprocessing.current_process()._identity[0]
    c = c_list[p_id % len(c_list)]
    p_name = ut.color_string('[Worker ' + str(p_id) + ']', type=c)
    print p_name, 'init random walk'
    assert isinstance(network, gt.Graph)
    net = network
    assert net.is_directed() is False
    vertices = list(net.vertices())
    pairs = defaultdict(set)
    print p_name, 'gen pairs'
    for i in xrange(num_pairs):
        src, targets = random.sample(vertices, 2)
        while targets in pairs[src]:
            src, targets = random.sample(vertices, 2)
        pairs[src].add(targets)
    shortest_distances = defaultdict(dict)
    print p_name, 'calc shortest distances'
    sys.stdout.flush()
    for src, targets in pairs.iteritems():
        sd = gt.topology.shortest_distance(net, src)
        for tar in targets:
            shortest_distances[src][tar] = sd[tar]
    print p_name, 'random walk'
    stretch = []
    num_success = 0
    for src, targets in pairs.iteritems():
        current_node = src
        for tar in targets:
            visited_nodes = set()
            sd = shortest_distances[src][tar]
            hops = 0
            while current_node != tar and (max_steps <= 0 or max_steps > hops):
                visited_nodes.add(current_node)
                hops += 1
                neighb = set(current_node.out_neighbours())
                if avoid_revisits:
                    neighb -= visited_nodes
                    if not neighb:
                        print p_name, 'Warn: all neighbours already visited'
                        neighb = set(current_node.out_neighbours())
                if tar in neighb:
                    current_node = tar
                else:
                    current_node = random.sample(neighb, 1)[0]
            if current_node == tar:
                num_success += 1
            stretch.append(hops / sd)
    print p_name, 'average stretch:', np.mean(stretch)
    try:
        sr = num_success/num_pairs
    except ZeroDivisionError:
        sr = 0
    return sr, stretch





