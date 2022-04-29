import random
import numpy as np


def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


class Dendrite:
    def __init__(self, from_nid):
        self.from_nid = from_nid
        self.weight = 2 * random.random() - 1
        self.trust = 1


class Neuron:
    def __init__(self, nid, is_real=False):
        self.nid = nid
        self.is_real = is_real
        self.dendrites = []
        self.output = 0
        self.backward = 0

    def mutate(self, neurons):
        population = [i for i in neurons.keys() if i != self.nid]
        random.shuffle(population)
        self.dendrites = [Dendrite(population[i]) for i in range(random.randint(2, 5))]

    def doForward(self, neurons):
        output = 0
        for d in self.dendrites:
            other = neurons[d.from_nid]
            output = output + (other.output * d.weight)
        # activation (relu)
        output = output if output > 0 else 0
        if self.is_real:
            self.backward = output - self.output
        else:
            self.output = output

    def doBackward(self, neurons):
        learning_rate = .5
        do_dzo = 1 if self.output > 0 else 0
        for d in self.dendrites:
            other = neurons[d.from_nid]
            db_dw = self.backward * do_dzo * other.output
            if not other.is_real:
                other.backward = other.backward + self.backward * do_dzo * d.weight
            # update
            d.weight = d.weight - learning_rate * db_dw
        self.backward = 0


class Brain:
    def __init__(self, world_size):
        self.world_size = world_size
        self.neurons = {i: Neuron(i, True) for i in range(world_size)}
        for i in range(world_size, 11):
            self.neurons[i] = Neuron(i)
        for nid in self.neurons:
            self.neurons[nid].mutate(self.neurons)

    def applyState(self, state):
        for i in range(self.world_size):
            self.neurons[i].output = state[i]

    def thinkOnce(self):
        population = [i for i in self.neurons.keys()]
        # output
        random.shuffle(population)
        for p in population:
            self.neurons[p].doForward(self.neurons)

        # backward
        random.shuffle(population)
        for p in population:
            self.neurons[p].doBackward(self.neurons)
