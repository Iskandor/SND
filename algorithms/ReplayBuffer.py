import itertools
import random
from collections import namedtuple, deque
from operator import itemgetter

import torch
import numpy as np

MDP_Transition = namedtuple('MDP_Transition', ('state', 'action', 'next_state', 'reward', 'mask'))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.memory = {}
        self.index = 0
        self.capacity_index = 0
        self.capacity = capacity

    def __len__(self):
        return self.capacity_index

    def indices(self, sample_size):
        ind = None
        if len(self) > sample_size:
            ind = random.sample(range(0, len(self)), sample_size)
        return ind


class GenericBuffer(object):
    def __init__(self, capacity, batch_size, n_env=1):
        self.keys = []
        self.n_env = n_env
        self.memory = {}
        self.index = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.transition = None

    def add(self, **kwargs):
        raise NotImplementedError

    def clear(self):
        pass


class GenericTrajectoryBuffer(GenericBuffer):
    def __len__(self):
        return self.index * self.n_env

    def memory_init(self, key, shape):
        steps_per_env = self.capacity // self.n_env
        shape = (steps_per_env, self.n_env,) + shape
        self.memory[key] = torch.zeros(shape)

    def add(self, **kwargs):
        if len(self.memory) == 0:
            for key in kwargs:
                self.keys.append(key)
                self.memory_init(key, tuple(kwargs[key].shape[1:]))
            self.transition = namedtuple('transition', self.keys)

        for key in kwargs:
            self.memory[key][self.index] = kwargs[key]

        self.index += 1

    def indices(self):
        ind = None
        if len(self) == self.capacity:
            ind = range(0, self.capacity)
        return ind

    def sample(self, indices, reshape_to_batch=True):
        if reshape_to_batch:
            values = [self.memory[k].reshape(-1, *self.memory[k].shape[2:]) for k in self.keys]
            result = self.transition(*values)
        else:
            values = [self.memory[k] for k in self.keys]
            result = self.transition(*values)

        return result

    def sample_batches(self, indices, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        values = [self.memory[k].reshape(-1, batch_size, *self.memory[k].shape[2:]) for k in self.keys]
        batch = self.transition(*values)
        return batch, self.capacity // batch_size

    def clear(self):
        self.index = 0


class GenericReplayBuffer(GenericBuffer):
    def __init__(self, capacity, batch_size, n_env=1):
        super().__init__(capacity, batch_size, n_env)
        self.capacity_index = 0

    def __len__(self):
        return self.index

    def indices(self, sample_size):
        ind = None
        if len(self) > sample_size:
            ind = random.sample(range(0, len(self)), sample_size)
        return ind

    def memory_init(self, key, shape):
        self.memory[key] = torch.zeros(shape)

    def add(self, **kwargs):
        if len(self.memory) == 0:
            for key in kwargs:
                self.keys.append(key)
                self.memory_init(key, (self.capacity,) + tuple(kwargs[key].shape[1:]))
            self.transition = namedtuple('transition', self.keys)

        for key in kwargs:
            self.memory[key][self.index:self.index + self.n_env] = kwargs[key][:]

        self.index += self.n_env
        if self.capacity_index < self.capacity:
            self.capacity_index += self.n_env
        if self.index == self.capacity:
            self.index = 0

    def sample(self, indices):
        values = [self.memory[k][indices] for k in self.keys]
        result = self.transition(*values)

        return result

    def sample_batches(self, indices, batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size

        original_batch = len(indices)

        values = []

        for k in self.keys:
            v = self.memory[k][indices]
            values.append(v.reshape(-1, batch_size, *v.shape[1:]))

        batch = self.transition(*values)
        return batch, original_batch // batch_size


if __name__ == '__main__':
    buffer = GenericReplayBuffer(100, 10, 2)

    for i in range(100):
        buffer.add(data=torch.rand(2, 4))
        indices = buffer.indices(10)
        if indices is not None:
            sample = buffer.sample_batches(indices, 10)
    pass