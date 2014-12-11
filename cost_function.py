from __future__ import division
from graph_tool.all import *
from tools.printing import print_f
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from collections import defaultdict
import random
import operator


class CostFunction():
    def __init__(self, graph, deg_weight=0.1, cos_weight=0.9, pairs=None):
        assert deg_weight + cos_weight == 1
        self.deg_weight = deg_weight
        self.cos_weight = cos_weight
        self.graph = graph
        self.print_f('Init cost function')
        self.print_f('\tget adjacency matrix')
        self.adj_mat = adjacency(graph).tocsc()
        self.print_f('\tgenerate pairs')
        if pairs is None:
            all_shortest_dist = True
            # format destination, array_of_sources
            self.pairs = [(i, np.array([j for j in xrange(self.adj_mat.shape[1]) if j != i])) for i in xrange(self.adj_mat.shape[0])]
        else:
            all_shortest_dist = False
            self.pairs = pairs
        self.print_f('\tgenerate degrees vector')
        deg_map = graph.degree_property_map('total')
        self.deg = csc_matrix((([deg_map[v] for v in graph.vertices()]), (range(graph.num_vertices()), [0] * graph.num_vertices())), shape=(graph.num_vertices(), 1))
        self.print_f('\tgenerate cosine similarity matrix')
        self.src = None
        self.src_n = None
        self.cos_sim = filter(lambda x: x[2] > 0, ((int(src), int(dest), self.calc_cossim(src, dest)) for src in graph.vertices() for dest in graph.vertices() if int(dest) >= int(src)))
        i, j, data = zip(*self.cos_sim)
        self.cos_sim = csc_matrix(coo_matrix((data, (j, i)), shape=(graph.num_vertices(), graph.num_vertices())))
        self.cos_sim = self.cos_sim + self.cos_sim.T
        self.cos_sim.setdiag(1)
        self.print_f('calc shortest distances for pairs')
        tmp_pairs = defaultdict(set)
        for dest, srcs in self.pairs:
            for i in srcs:
                tmp_pairs[i].add(dest)
        shortest_distances = defaultdict(lambda: defaultdict(int))
        if not all_shortest_dist:
            # shortest_distances = [(src, dest, shortest_distance(graph.vertex(src), graph.vertex(dest))) for src, dests in tmp_pairs.iteritems() for dest in dests]
            pass
        else:
            s_map = shortest_distance(graph)
            for src, dests in tmp_pairs.iteritems():
                for dest in dests:
                    shortest_distances[dest][src] = s_map[graph.vertex(src)][dest]
        # TODO: is there a way to include all shortest distance neigbours?
        best_neighbours = defaultdict(lambda: dict())
        for src, dests in tmp_pairs.iteritems():
            for dest in dests:
                sd = shortest_distances[dest]
                best_n = set()
                best_sd = 10000000000000
                for n in graph.vertex(src).out_neighbours():
                    n_sd = sd[int(n)]
                    if n_sd < best_sd:
                        best_sd = n_sd
                        best_n = {int(n)}
                    elif n_sd == best_sd:
                        best_n.add(int(n))
                best_n = random.sample(best_n, 1)[0]
                best_neighbours[src][dest] = best_n
        data, i, j = zip(*[(best_neighbours[i][j], i, j) for i in xrange(graph.num_vertices()) for j in xrange(graph.num_vertices()) if i != j])
        self.best_neighbours = csc_matrix((data, (i, j)))

    def calc_cossim(self, src, dest):
        if src == dest:
            return 1.0
        if self.src != src:
            self.src_n = set(src.all_neighbours())
            self.src = src
        dest_n = set(dest.all_neighbours())
        return len(self.src_n & dest_n) / np.sqrt(len(self.src_n) * len(dest_n))

    @staticmethod
    def print_f(*args, **kwargs):
        kwargs.update({'class_name': 'CostFunction'})
        print_f(*args, **kwargs)

    def calc_cost(self, v=None):
        self.print_f('calc cost')
        cost = 0
        for dest, srcs in self.pairs:
            # print 'cos'
            # print cos.T.todense()
            # print srcs[0], 'neighbours'
            # print self.adj_mat[:, srcs[0]].T.todense()
            # print 'product'
            # print cos.multiply(self.adj_mat[:, srcs])[:,0] result of source nummer 0
            cos = self.cos_sim[:, dest].multiply(self.adj_mat[:, srcs])
            cos = cos.multiply(1 / cos.sum(axis=0))
            deg = self.deg.multiply(self.adj_mat[:, srcs])
            deg = deg.multiply(1 / deg.sum(axis=0))
            prop = deg * self.deg_weight + cos * self.cos_weight
            part_cost = np.nansum((prop * self.best_neighbours[srcs, :]))
            cost += part_cost
        self.print_f('cost:', cost)
        return cost

