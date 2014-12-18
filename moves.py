from __future__ import division
from abc import abstractmethod
import math
from tools.printing import print_f
import random


class Mover(object):
    def __init__(self):
        self.moves_counter = 0

    @abstractmethod
    def move(self, vector):
        self.moves_counter += 1

    @abstractmethod
    def reduce_step_size(self):
        self.print_f('reduce step size')
        pass

    @staticmethod
    def print_f(*args, **kwargs):
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
    def __init__(self):
        Mover.__init__(self)
        pass

    def move(self, vector):
        super(MoveTravelSM, self).move(vector)
        p = random.uniform(0.0, 1.0)
        vector_len = len(vector)
        if p < 0.2:
            i = random.randint(0, vector_len / 2)
            vector = vector[i:] + vector[:i]
            i = random.randint(0, vector_len / 2)
            a = vector[:i]
            a.reverse()
            vector = a + vector[i:]
        elif p < 0.6:
            i = random.randint(0, vector_len - 1)
            a = vector.pop(i)
            j = random.randint(0, len(vector) - 1)
            vector.insert(j, a)
        else:
            i = random.randint(0, vector_len)
            j = random.randint(0, vector_len)
            vector[i], vector[j] = vector[j], vector[i]
        return vector

    def reduce_step_size(self):
        super(MoveTravelSM, self).reduce_step_size()
        pass






