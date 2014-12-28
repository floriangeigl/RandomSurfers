from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
from abc import abstractmethod
from cost_function import CostFunction
from moves import Mover
import random
import copy
import numpy as np
from tools.printing import print_f
from itertools import cycle


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
        self.cost_history = []

    def draw_cost_history(self, filename='output/cost.png', compare_dict=None):
        colors = ['blue', 'red', 'yellow']
        plt.plot(self.cost_history, lw=3, c='black', label='cost', alpha=0.5)
        if compare_dict is not None:
            for idx, (name, vals) in enumerate(compare_dict.iteritems()):
                c = colors[idx % len(colors)]
                if hasattr(vals, '__iter__'):
                    plt.plot(vals, lw=2, label=name, c=c, alpha=0.4)
                else:
                    plt.axhline(y=vals, label=name, c=c, alpha=0.4)
        plt.legend(loc='best')
        plt.savefig(filename, dpi=150)
        plt.close('all')

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
        Optimizer.__init__(self, cost_function, mover, init_nodes_ranking, known=known, max_runs=max_runs, reduce_step_after_fails=reduce_step_after_fails, reduce_step_after_accepts=reduce_step_after_accepts, *args, **kwargs)
        self.beta = beta
        self.prob_history = []

    def optimize(self):
        self.accepts = 0
        self.fails = 0
        current_ranking = copy.copy(self.init_ranking)
        current_cost = self.cf.calc_cost(current_ranking)
        best_cost = current_cost
        best_ranking = current_ranking
        perc = -10
        for i in xrange(self.runs):
            c_perc = int((i / self.runs) * 100)
            if self.verbose >= 3:
                self.print_f('run:', i, '(', c_perc, '% )', verbose=3)
            elif c_perc >= perc + 10:
                perc = c_perc
                self.print_f('run:', i, '(', c_perc, '% )')
            new_ranking = self.mv.move(current_ranking)
            new_cost = self.cf.calc_cost(new_ranking)
            self.cost_history.append(new_cost)
            accept_prob = np.exp(- self.beta * (current_cost - new_cost))
            self.prob_history.append(accept_prob)
            if random.uniform(0.0, 1.0) < accept_prob:
                self.accepts += 1
                current_cost = new_cost
                current_ranking = new_ranking
                if new_cost > best_cost:
                    best_cost = new_cost
                    best_ranking = copy.copy(current_ranking)
            else:
                self.fails += 1
            if 0 < self.reduce_after_a <= self.accepts or 0 < self.reduce_after_f <= self.fails:
                self.mv.reduce_step_size()
                self.beta *= 1.05
                self.accepts = 0
                self.fails = 0
        return best_ranking, best_cost
