from abc import abstractmethod
from cost_function import CostFunction
from moves import Mover


class Optimizer(object):
    def __init__(self, cost_function, mover, nodes_ranking, known=0.1, max_runs=100, reduce_step_after_fails=0):
        assert isinstance(cost_function, CostFunction)
        assert isinstance(mover, Mover)
        self.cf = cost_function
        self.mv = mover
        self.nodes_r = nodes_ranking
        self.known = known
        self.reduce_after = reduce_step_after_fails
        self.runs = max_runs
        self.fails = 0

    def optimize(self):
        num_known_nodes = int(round(len(self.nodes_r) * self.known))
        best_cost = -1
        new_ranking = self.nodes_r
        for i in xrange(self.runs):
            known_nodes = new_ranking[:num_known_nodes]
            current_cost = self.cf.calc_cost(known_nodes)
            if current_cost > best_cost:
                best_cost = current_cost
                self.nodes_r = new_ranking
            else:
                self.fails += 1
            if self.fails > self.reduce_after:
                self.mv.reduce_step_size()
            new_ranking = self.mv.move(self.nodes_r)
