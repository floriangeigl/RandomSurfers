from __future__ import division
import unittest
import random

from tools.gt_tools import GraphGenerator
from old_version import moves, cost_function
from old_version.optimizer import Optimizer, SimulatedAnnealing
from utils import *


class TestMover(unittest.TestCase):
    def test_MoveSwapper(self):
        mover = moves.MoveSwapper(size=0.1)
        v = range(10)
        v_old = v
        v = mover.move(v)
        self.assertEqual(v_old, range(10))
        self.assertEqual(v, [1, 0] + range(2, 10))
        mover = moves.MoveSwapper(size=0.1, upper=False)
        v = range(10)
        v = mover.move(v)
        self.assertEqual(v, range(8) + [9, 8])

    def test_CostFunction(self):
        network = GraphGenerator().create_karate_graph()
        cf = cost_function.CostFunction(network)
        cf.calc_cost()
        random_known_nodes = random.sample(range(network.num_vertices()), int(network.num_vertices() * 0.1))
        cf.calc_cost(random_known_nodes)

    def test_simple_optimizer(self):
        network = GraphGenerator(200)
        network = network.create_random_graph()
        cf = cost_function.CostFunction(network, pairs_reduce=0.1)
        mover = moves.MoveShuffle(size=1)
        all_nodes = range(network.num_vertices())
        random.shuffle(all_nodes)
        opt = Optimizer(cf, mover, all_nodes, known=0.1, max_runs=100, reduce_step_after_fails=10)
        opt.optimize()

    def test_simulated_annealing(self):
        network = Graph(directed=False)
        vertices = [network.add_vertex() for i in range(5)]
        edges = [network.add_edge(vertices[0], v) for v in vertices[1:]]
        edges += [network.add_edge(vertices[1], v) for v in vertices[2:]]
        # star-like network with two main nodes: v0 connected to all, v1 connected to all without v0
        init_nodes_ranking = range(network.num_vertices())
        random.shuffle(init_nodes_ranking)
        cf = cost_function.CostFunction(network, target_reduce=1)
        mover = moves.MoveTravelSM()
        print 'init ranking:', init_nodes_ranking
        opt = SimulatedAnnealing(cf, mover, init_nodes_ranking=init_nodes_ranking, max_runs=200, reduce_step_after_fails=0, reduce_step_after_accepts=100)
        ranking, cost = opt.optimize()
        print 'runs:', opt.runs
        print 'best ranking', ranking
        print 'cost:', cost
        assert ranking[0] == 0
        assert ranking[1] == 1


    def test_CostFunctionSpeed(self):
        network = GraphGenerator(1000)
        network = network.create_random_graph()
        cf = cost_function.CostFunction(network, target_reduce=0.1)
        cf.calc_cost()
        random_known_nodes = range(network.num_vertices())
        for i in range(1, 11):
            num_known_nodes = int(round(network.num_vertices() * (1 / (i * 10))))
            random_known_nodes = random.sample(random_known_nodes, num_known_nodes)
            cf.calc_cost(random_known_nodes)

    def test_Ranking(self):
        ranking = range(10) + list(reversed(range(10, 20)))
        weights = list(reversed(range(len(ranking))))
        random.shuffle(weights)
        weights = np.array(weights).astype(float)
        weights /= weights.max()
        print ranking
        print weights
        df = get_ranking_df(ranking, weights)
        g = Graph()
        vertices = [g.add_vertex() for i in range(len(ranking))]
        for v1 in vertices:
            for v2 in vertices[int(v1):]:
                g.add_edge(v1, v2)
        deg_map = g.degree_property_map('total')
        df['deg'] = get_ranking(deg_map)
        assert range(len(ranking)) == list(df['deg'])
        assert list(df['ranked_vertex']) == ranking
        assert list(df['values']) == list(weights)

