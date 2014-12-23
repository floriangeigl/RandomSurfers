from __future__ import division
import unittest
import moves
import copy
import cost_function
from tools.gt_tools import GraphGenerator
import timeit
import random
from optimizer import Optimizer, SimulatedAnnealing
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
        network = GraphGenerator(50)
        network = network.create_random_graph()
        cf = cost_function.CostFunction(network, pairs_reduce=0.1)
        mover = moves.MoveTravelSM()
        all_nodes = range(network.num_vertices())
        random.shuffle(all_nodes)
        opt = SimulatedAnnealing(cf, mover, all_nodes, known=0.1, max_runs=1000, reduce_step_after_fails=0, reduce_step_after_accepts=100)
        ranking, cost = opt.optimize()
        print 'runs:', opt.runs
        print 'best ranking', ranking
        print 'cost:', cost

    def test_CostFunctionSpeed(self):
        network = GraphGenerator(1000)
        network = network.create_random_graph()
        cf = cost_function.CostFunction(network, pairs_reduce=0.1)
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
