from __future__ import division
from abc import abstractmethod
from cost_function import CostFunction
from moves import Mover
import random
import numpy as np
from tools.printing import print_f


class Optimizer(object):
    def __init__(self, cost_function, mover, init_nodes_ranking, known=0.1, max_runs=100, reduce_step_after_fails=0, reduce_step_after_accepts=0, verbose=2, *args, **kwargs):
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
        self.accepts = 0
        self.verbose = verbose

    def optimize(self):
        num_known_nodes = int(round(len(self.init_ranking) * self.known))
        best_cost = None
        new_ranking = self.init_ranking
        for i in xrange(self.runs):
            self.print_f('run:', i, '(', (i / self.runs) * 100, '% )')
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

    def print_f(self, *args, **kwargs):
        if 'verbose' not in kwargs or kwargs['verbose'] <= self.verbose:
            kwargs.update({'class_name': 'Optimizer'})
            print_f(*args, **kwargs)


class SimulatedAnnealing(Optimizer):
    def __init__(self, cost_function, mover, init_nodes_ranking, known=0.1, max_runs=100, reduce_step_after_fails=0, reduce_step_after_accepts=0, beta=1.0, *args, **kwargs):
        Optimizer.__init__(self, cost_function, mover, init_nodes_ranking, known=known, max_runs=max_runs, reduce_step_after_fails=reduce_step_after_fails, reduce_step_after_accepts=reduce_step_after_accepts)
        self.beta = beta

    def optimize(self):
        self.accepts = 0
        self.fails = 0
        current_init_ranking = self.init_ranking
        num_known_nodes = int(round(len(current_init_ranking) * self.known))
        new_ranking = current_init_ranking
        current_ranking = current_init_ranking
        known_nodes = new_ranking[:num_known_nodes]
        cost = self.cf.calc_cost(known_nodes)
        best_cost = cost
        best_ranking = None
        for i in xrange(self.runs):
            self.print_f('run:', i, '(', (i / self.runs) * 100, '% )')
            known_nodes = new_ranking[:num_known_nodes]
            current_cost = self.cf.calc_cost(known_nodes)
            if random.uniform(0.0, 1.0) < np.exp(- self.beta * (current_cost - best_cost)):
                self.accepts += 1
                current_ranking = new_ranking
                if current_cost > best_cost:
                    best_cost = current_cost
                    best_ranking = current_ranking
            else:
                self.fails += 1
            new_ranking = self.mv.move(current_ranking)
            if 0 < self.accepts <= self.reduce_after_a or 0 < self.fails <= self.reduce_after_f:
                self.mv.reduce_step_size()
                self.beta *= 1.005
                self.accepts = 0
                self.fails = 0
        return best_ranking, best_cost
