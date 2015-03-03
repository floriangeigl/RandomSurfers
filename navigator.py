from __future__ import division
import graph_tool as gt
import random
import operator
from collections import defaultdict
import numpy as np
import os

def random_walk(network, max_steps, avoid_revisits=True, num_pairs=1000):
    print 'init random walk'
    p_name = '[' + str(os.getpid()) + ']'
    assert isinstance(network, gt.Graph)
    net = network
    assert net.is_directed() is False
    vertices = [net.vertices()]
    pairs = defaultdict(set)
    print '\t', p_name, 'gen pairs'
    for i in xrange(num_pairs):
        src, target = random.sample(vertices, 2)
        while target in pairs[src]:
            src, target = random.sample(vertices, 2)
        pairs[src].add(target)
    shortest_distances = defaultdict(dict)
    print '\t', p_name, 'calc shortest distances'
    for src, targets in pairs.iteritems():
        sd = gt.topology.shortest_distance(net, src)
        for t in targets:
            shortest_distances[src][t] = sd[t]
    print '\t', p_name, 'random walk'
    stretch = []
    num_success = 0
    for src, target in pairs:
        current_node = src
        visited_nodes = set()
        sd = shortest_distances[src][target]
        hops = 0
        while current_node != target and (max_steps <= 0 or max_steps > hops):
            visited_nodes.add(current_node)
            hops += 1
            neighb = set(current_node.out_neighbours())
            if avoid_revisits:
                neighb -= visited_nodes
                if not neighb:
                    print '\t', p_name, 'Warn: all neighbours already visited'
                    neighb = set(current_node.out_neighbours())
            if target in neighb:
                current_node = target
            else:
                current_node = random.sample(neighb, 1)[0]
        if current_node == target:
            num_success += 1
        stretch.append(hops / sd)
    print '\t', p_name, 'average stretch:', np.mean(stretch)
    try:
        sr = num_success/num_pairs
    except ZeroDivisionError:
        sr = 0
    return sr, stretch






