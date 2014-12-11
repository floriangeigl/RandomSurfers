import unittest
import moves
import copy
import cost_function
from tools.gt_tools import GraphGenerator


class TestMover(unittest.TestCase):
    def test_MoveSwapper(self):
        mover = moves.MoveSwapper(swap_size=0.1)
        v = range(10)
        v_old = v
        v = mover.move(v)
        self.assertEqual(v_old, range(10))
        self.assertEqual(v, [1, 0] + range(2, 10))
        mover = moves.MoveSwapper(swap_size=0.1, upper=False)
        v = range(10)
        v = mover.move(v)
        self.assertEqual(v, range(8) + [9, 8])

    def test_CostFunction(self):
        network = GraphGenerator().create_karate_graph()
        cf = cost_function.CostFunction(network)
        cf.calc_cost()
