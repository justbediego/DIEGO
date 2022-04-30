import random
import numpy as np


def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


def activation(x):
    return sigmoid(x)


def dActivation(x):
    tmp = sigmoid(x)
    return tmp * (1 - tmp)


class Dendrite:
    def __init__(self, from_nid):
        self.from_nid = from_nid
        self.yes, self.no = random.choices([i for i in range(10)], k=2)

    def getWeight(self):
        # 1 to -1
        return (self.yes - self.no) / (self.yes + self.no)


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
        self.dendrites = [Dendrite(population[i]) for i in range(random.randint(3, 3))]

    def doForwardBackward(self, neurons):
        # forward
        z = 0
        for d in self.dendrites:
            other = neurons[d.from_nid]
            z = z + other.output * d.getWeight()
        o = activation(z)
        if self.is_real:
            self.backward = self.output - o
        else:
            self.output = o
        # backward
        do_dzo = dActivation(z)
        for d in self.dendrites:
            other = neurons[d.from_nid]
            db_dw = self.backward * do_dzo * other.output
            if not other.is_real:
                other.backward = other.backward + self.backward * do_dzo * d.getWeight()
            # update
            if db_dw > 0:
                d.yes = d.yes + db_dw
            else:
                d.no = d.no + db_dw
        self.backward = 0


class Brain:
    def __init__(self, world_size):
        self.world_size = world_size
        self.neurons = {i: Neuron(i, True) for i in range(world_size)}
        for i in range(world_size, 3):
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
            self.neurons[p].doForwardBackward(self.neurons)
