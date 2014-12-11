import unittest
import moves
import copy


class TestMover(unittest.TestCase):
    def test_MoveSwapper(self):
        mover = moves.MoveSwapper(swap_size=0.5)
        v = range(10)
        v_old = copy.copy(v)
        mover.move(v)
        print v, v_old