from __future__ import division
from abc import abstractmethod
import copy
from tools.printing import print_f
import random
import numpy as np


class Mover(object):
    def __init__(self, verbose=2):
        self.moves_counter = 0
        self.verbose = verbose

    @abstractmethod
    def move(self, vector):
        self.moves_counter += 1

    @abstractmethod
    def reduce_step_size(self):
        self.print_f('reduce step size', verbose=2)
        pass

    def print_f(self, *args, **kwargs):
        if 'verbose' not in kwargs or kwargs['verbose'] <= self.verbose:
            kwargs.update({'class_name': 'Mover'})
            print_f(*args, **kwargs)


class MoveSwapper(Mover):
    def __init__(self, size=0.4, upper=True, reduce_function=lambda x: x / 2):
        Mover.__init__(self)
        self.swap_size = size
        assert size <= 0.5
        self.upper = upper
        self.reduce_fun = reduce_function

    def move(self, vector):
        super(MoveSwapper, self).move(vector)
        vector = copy.copy(vector)
        num_elements = int(round(self.swap_size * len(vector)))
        if self.upper:
            vector = vector[num_elements:num_elements * 2] + vector[:num_elements] + vector[num_elements * 2:]
        else:
            vector = vector[:-num_elements * 2] + vector[-num_elements:] + vector[-num_elements * 2:-num_elements]
        return vector

    def reduce_step_size(self):
        super(MoveSwapper, self).reduce_step_size()
        self.swap_size = self.reduce_fun(self.swap_size)


class MoveShuffle(Mover):
    def __init__(self, size=1, upper=True, reduce_function=lambda x: x / 2):
        Mover.__init__(self)
        self.shuffle_size = size
        self.reduce_fun = reduce_function
        self.upper = upper

    def move(self, vector):
        super(MoveShuffle, self).move(vector)
        vector = copy.copy(vector)
        num_elements = int(round(self.shuffle_size * len(vector)))
        if self.upper:
            shuffle_part = vector[:num_elements]
            static_part = vector[num_elements:]
        else:
            shuffle_part = vector[num_elements:]
            static_part = vector[:num_elements]
        random.shuffle(shuffle_part)
        vector = shuffle_part + static_part
        return vector

    def reduce_step_size(self):
        super(MoveShuffle, self).reduce_step_size()
        self.shuffle_size = self.reduce_fun(self.shuffle_size)


class MoveTravelSM(Mover):
    def __init__(self, *args, **kwargs):
        Mover.__init__(self, *args, **kwargs)
        self.big_move_prob = 0.2
        self.small_moves_prob = 0.4

    def move(self, vector):
        super(MoveTravelSM, self).move(vector)
        vector = copy.copy(vector)
        p = random.uniform(0.0, 1.0)
        rand_int = np.random.randint
        len_vec = len(vector)
        if p < self.big_move_prob:
            half_idx = int(len_vec / 2)
            i = rand_int(half_idx)
            vector = vector[i:] + vector[:i]
            i = rand_int(half_idx)
            a = vector[:i]
            a.reverse()
            vector = a + vector[i:]
        elif p < self.big_move_prob + self.small_moves_prob:
            i, j = rand_int(len_vec, size=2)
            while i == j:
                i, j = rand_int(len_vec, size=2)
            vector[i], vector[j] = vector[j], vector[i]
        else:
            i = rand_int(len_vec)
            a = vector[i]
            del vector[i]
            j = rand_int(len_vec - 1)
            vector.insert(j, a)
        return vector

    def reduce_step_size(self):
        super(MoveTravelSM, self).reduce_step_size()
        self.big_move_prob *= 0.75
        self.small_moves_prob = (1 - self.big_move_prob) / 2






