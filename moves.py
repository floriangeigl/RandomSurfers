from __future__ import division
from abc import abstractmethod
import math


class Mover(object):
    def __init__(self):
        self.moves_counter = 0

    @abstractmethod
    def move(self, vector):
        self.moves_counter += 1

    @abstractmethod
    def reduce_step_size(self):
        pass


class MoveSwapper(Mover):
    def __init__(self, swap_size=0.4, upper=True, reduce_function=lambda x: x / 2):
        Mover.__init__(self)
        self.swap_size = swap_size
        assert swap_size <= 0.5
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
        self.swap_size = self.reduce_fun(self.swap_size)





