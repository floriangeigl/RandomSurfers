from abc import abstractmethod
from cost_function import CostFunction
from moves import Mover
import random
import math


class Optimizer(object):
    def __init__(self, cost_function, mover, init_nodes_ranking, known=0.1, max_runs=100, reduce_step_after_fails=0, reduce_step_after_accepts=0, *args, **kwargs):
        assert isinstance(cost_function, CostFunction)
        assert isinstance(mover, Mover)
        self.cf = cost_function
        self.mv = mover
        self.init_ranking = init_nodes_ranking
        self.known = known
        self.reduce_after_f = reduce_step_after_fails
        self.reduce_after_a = reduce_step_after_accepts
        self.runs = max_runs
        self.fails = 0
        self.best_ranking = None
        self.accepts = 0

    def optimize(self):
        num_known_nodes = int(round(len(self.init_ranking) * self.known))
        best_cost = None
        new_ranking = self.init_ranking
        for i in xrange(self.runs):
            known_nodes = new_ranking[:num_known_nodes]
            current_cost = self.cf.calc_cost(known_nodes)
            if best_cost is None or current_cost > best_cost:
                best_cost = current_cost
                self.init_ranking = new_ranking
                self.accepts += 1
            else:
                self.fails += 1
            if self.fails > self.reduce_after_f:
                self.mv.reduce_step_size()
            new_ranking = self.mv.move(self.init_ranking)


class SimulatedAnnealing(Optimizer):
    def __init__(self, cost_function, mover, init_nodes_ranking, known=0.1, max_runs=100, reduce_step_after_fails=0, reduce_step_after_accepts=0, beta=1.0, *args, **kwargs):
        Optimizer.__init__(self, cost_function, mover, init_nodes_ranking, known=0.1, max_runs=100, reduce_step_after_fails=0, reduce_step_after_accepts=0)
        self.beta = beta

    def optimize(self):
        best_cost = None
        self.accepts = 0
        self.fails = 0
        num_known_nodes = int(round(len(self.init_ranking) * self.known))
        new_ranking = self.init_ranking
        current_ranking = self.init_ranking
        known_nodes = new_ranking[:num_known_nodes]
        cost = self.cf.calc_cost(known_nodes)
        for i in xrange(self.runs):
            known_nodes = new_ranking[:num_known_nodes]
            current_cost = self.cf.calc_cost(known_nodes)
            if random.uniform(0.0, 1.0) < math.exp(- self.beta * (current_cost - cost)):
                self.accepts += 1
                current_ranking = new_ranking
                if best_cost is None or current_cost > best_cost:
                    best_cost = current_cost
                    self.best_ranking = current_ranking
                    self.init_ranking = new_ranking
            else:
                self.fails += 1
            new_ranking = self.mv.move(current_ranking)
            if 0 < self.accepts <= self.reduce_after_a or 0 < self.fails <= self.reduce_after_f:
                self.mv.reduce_step_size()
                self.beta *= 1.005
                self.accepts = 0
                self.fails = 0
