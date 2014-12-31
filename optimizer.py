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
import pandas as pd


class Optimizer(object):
    def __init__(self, cost_function, mover, init_nodes_ranking, known=0.1, runs=100, reduce_step_after_fails=0, reduce_step_after_accepts=0, verbose=2, *args, **kwargs):
        assert isinstance(cost_function, CostFunction)
        assert isinstance(mover, Mover)
        self.cf = cost_function
        self.mv = mover
        self.init_ranking = init_nodes_ranking
        self.known = known
        self.reduce_after_f = reduce_step_after_fails
        self.reduce_after_a = reduce_step_after_accepts
        self.runs = runs
        self.fails = 0
        self.accepts = 0
        self.verbose = verbose
        self.cost_history = []

    def draw_cost_history(self, filename='output/cost.png', compare_dict=None):
        colors = ['blue', 'red', 'yellow']
        f, ax = plt.subplots()
        ax.plot(self.cost_history, lw=1, c='black', label='cost', alpha=0.8)
        df = pd.DataFrame(columns=['rolling mean'], data=self.cost_history)
        df['rolling mean'] = pd.rolling_mean(df['rolling mean'], window=100)
        df.plot(ax=ax, lw=3, c='green')
        if compare_dict is not None:
            for idx, (name, vals) in enumerate(compare_dict.iteritems()):
                c = colors[idx % len(colors)]
                if hasattr(vals, '__iter__'):
                    plt.plot(vals, lw=2, label=name, c=c, alpha=0.6)
                else:
                    plt.axhline(y=vals, lw=2, label=name, c=c, alpha=0.6)
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
    def __init__(self, cost_function, mover, init_nodes_ranking, known=0.1, runs=100, reduce_step_after_fails=0, reduce_step_after_accepts=0, beta=1.0, runs_per_temp=100, *args, **kwargs):
        Optimizer.__init__(self, cost_function, mover, init_nodes_ranking, known=known, runs=runs, reduce_step_after_fails=reduce_step_after_fails, reduce_step_after_accepts=reduce_step_after_accepts, *args, **kwargs)
        self.beta = beta
        self.prob_history = []
        self.beta_history = {}
        self.accept_deny_history = []
        self.runs_per_temp = runs_per_temp

    def get_acceptance_rate(self, n_last=None):
        if n_last is None:
            return np.mean(self.accept_deny_history)
        else:
            return np.mean(self.accept_deny_history[-n_last:])

    def find_beta(self, target_acceptance_rate=.5, runs=None, ranking=None, init_beta=None):
        beta = self.beta if init_beta is None else init_beta
        runs = self.runs_per_temp if runs is None else runs
        orig_cf_reduce = (self.cf.source_reduce, self.cf.target_reduce)
        self.cf.source_reduce = 0.1
        self.cf.target_reduce = 0.01
        last_accept_rate = 1
        while True:
            current_ranking = copy.copy(self.init_ranking) if ranking is None else copy.copy(ranking)
            current_cost = self.cf.calc_cost(current_ranking)
            accept_rate = []
            for i in range(runs):
                new_ranking = self.mv.move(current_ranking)
                new_cost = self.cf.calc_cost(new_ranking)
                accept_prob = np.exp(- beta * (current_cost - new_cost)) if new_cost < current_cost else 1
                if random.uniform(0.0, 1.0) <= accept_prob:
                    accept_rate.append(1)
                    current_cost = new_cost
                    current_ranking = new_ranking
                else:
                    accept_rate.append(0)
            accept_rate = np.mean(accept_rate)
            self.print_f('find beta: current beta: ', beta, '||current accept rate:', accept_rate, '||target:', target_acceptance_rate)
            if accept_rate > target_acceptance_rate:
                beta *= 1.1
                last_accept_rate = accept_rate
            else:
                if abs(last_accept_rate - target_acceptance_rate) < abs(accept_rate - target_acceptance_rate):
                    beta /= 1.1
                break
        self.cf.source_reduce, self.cf.target_reduce = orig_cf_reduce
        self.print_f('best beta for ', target_acceptance_rate, 'accept rate:', beta)
        return beta

    def optimize(self, max_runs=None):
        self.accepts = 0
        self.fails = 0
        self.accept_deny_history = []
        self.beta_history = dict()
        self.prob_history = []
        self.cost_history = []
        self.print_f('find good init beta')
        # beta = self.find_beta(target_acceptance_rate=0.8)
        # beta_07 = self.find_beta(target_acceptance_rate=0.7, init_beta=beta)
        #self.print_f('beta accept rate 0.8:', beta)
        #self.print_f('beta accept rate 0.7:', beta_07)
        #beta_fac = beta_07 / beta
        #self.print_f('beta fac:', beta_fac)
        beta = self.beta
        beta_fac = 1.5
        current_ranking = copy.copy(self.init_ranking)
        current_cost = self.cf.calc_cost(current_ranking)
        best_cost = current_cost
        best_ranking = current_ranking
        best_cost_history = []
        run = 0
        # for run in xrange(self.runs):
        self.print_f('init cost:', best_cost, '||init beta:', beta, )
        init_cost = best_cost
        while True:
            run += 1
            if run % self.runs_per_temp == 0:
                accept_rate = np.mean(self.accept_deny_history[-self.runs_per_temp:])
                self.print_f('run:', run, '||best cost:', best_cost, '|| improvement:', (best_cost / init_cost) - 1, '||beta:', beta, '||acceptance rate:', accept_rate)
                beta *= beta_fac
                self.beta_history[run] = beta
                current_cost = best_cost
                current_ranking = copy.copy(best_ranking)
                if accept_rate < 0.5:
                    self.mv.reduce_step_size()
                if len(best_cost_history) >= self.runs_per_temp * 10 and len(set(best_cost_history[-self.runs_per_temp * 10:])) == 1:
                    break
            if max_runs is not None and run > max_runs:
                break
            new_ranking = self.mv.move(current_ranking)
            new_cost = self.cf.calc_cost(new_ranking)
            self.cost_history.append(new_cost)
            accept_prob = np.exp(- beta * (current_cost - new_cost)) if new_cost < current_cost else 1
            self.prob_history.append(accept_prob)
            if random.uniform(0.0, 1.0) <= accept_prob:
                self.accept_deny_history.append(1)
                current_cost = new_cost
                current_ranking = new_ranking
                if new_cost > best_cost:
                    best_cost = new_cost
                    best_ranking = copy.copy(current_ranking)
            else:
                self.accept_deny_history.append(0)
            best_cost_history.append(best_cost)
        return best_ranking, best_cost
