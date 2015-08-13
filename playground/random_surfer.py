import graph_tool.all as gt
import random
import numpy as np
import multiprocessing as mp
import traceback

def rand_surf(g, weights, hobs):
    try:
        visits = g.new_vertex_property('int')
        try:
            weights = g.ep[weights]
        except KeyError:
            weights = None
        vertices = list(g.vertices())
        degrees = np.array(g.degree_property_map('in').a)
        rand_num = random.random() * degrees.sum()
        weights_sum = 0.
        for v, w in zip(vertices, degrees):
            weights_sum += w
            if weights_sum >= rand_num:
                current_v = v
                break

        current_v = random.sample(list(g.vertices()), k=1)[0]
        for i in xrange(hobs):
            visits[current_v] += 1
            out_neighbours = list(current_v.out_neighbours())
            if weights is None:
                current_v = random.sample(out_neighbours, 1)[0]
            else:
                out_weights = np.array(list(map(lambda x: weights[g.edge(current_v, x)], out_neighbours)))
                rand_num = random.random() * out_weights.sum()
                weights_sum = 0.
                for v, w in zip(out_neighbours, out_weights):
                    weights_sum += w
                    if weights_sum >= rand_num:
                        current_v = v
                        break
        return np.array(visits.a).astype('float')
    except:
        print traceback.format_exc()
        return 'error'


class RandomSurfer():
    def __init__(self, network, weights=None):
        self.g = network
        self.directed = self.g.is_directed()
        self.g.purge_vertices()
        self.g.purge_edges()
        self.g.clear_filters()
        if weights is not None and len(weights.a) == self.g.num_vertices():
            self.weights = self.g.new_edge_property('float')
            for e in self.g.edges():
                self.weights[e] = weights[e.target()]
        else:
            self.weights = weights
        assert np.all(gt.label_largest_component(self.g).a)

    def surf(self, num_hops=None, processes=1):
        if num_hops is None:
            num_hops = self.g.num_edges() * 10000
        # print 'iter:', num_hops
        results = list()
        result_append = results.append
        worker_pool = mp.Pool(processes=processes)
        num_hops = int(num_hops / processes) + 1
        if self.weights is not None:
            self.g.ep['w'] = self.weights
        for i in xrange(processes):
            worker_pool.apply_async(rand_surf, args=(self.g.copy(), 'w', num_hops), callback=result_append)
        worker_pool.close()
        worker_pool.join()
        stat_dist = sum(results)
        stat_dist /= stat_dist.sum()
        return stat_dist
