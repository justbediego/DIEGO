import random
import numpy as np


def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


def activation(x):
    return 1 if x > 0 else 0


def dActivation(x):
    return 1 if x > 0 else 0


class Dendrite:
    def __init__(self, from_nid, yes_no=None):
        self.from_nid = from_nid
        if yes_no is not None:
            self.yes, self.no = yes_no
        else:
            self.yes, self.no = random.choices([i for i in range(1, 10)], k=2)
        # self.weight = 2 * random.random() - 1

    def getWeight(self):
        # 1 to -1
        return (self.yes - self.no) / (self.yes + self.no)
        # return self.weight

    def increaseWeight(self, amount):
        if amount > 0:
            self.yes = self.yes + 1
        elif amount < 0:
            self.no = self.no + 1
        # self.weight = self.weight + amount


class Neuron:
    def __init__(self, nid, is_real=False, dendrites=[]):
        self.nid = nid
        self.is_real = is_real
        self.dendrites = dendrites
        self.output = 0
        self.backward = 0

    def mutate(self, neurons):
        population = [i for i in neurons.keys() if i != self.nid]
        random.shuffle(population)
        self.dendrites = [Dendrite(population[i]) for i in range(random.randint(2, 4))]

    def doForward(self, neurons):
        if self.is_real:
            self.backward = self.output
        else:
            z = 0
            for d in self.dendrites:
                other = neurons[d.from_nid]
                z = z + other.output * d.getWeight()
            self.output = activation(z)

    def doBackward(self, neurons):
        for d in self.dendrites:
            other = neurons[d.from_nid]
            if not other.is_real:
                other.backward = other.backward + self.backward * d.getWeight()
            # update
            diff = 1 - 2 * abs(other.output - self.backward)
            d.increaseWeight(diff)
        self.backward = 0


class Brain:
    def __init__(self, world_size, file_name=None):
        self.world_size = world_size
        if file_name is not None:
            self.loadBrain(file_name)
        else:
            self.neurons = {i: Neuron(i, True) for i in range(world_size)}
            for i in range(world_size, 30):
                self.neurons[i] = Neuron(i)
            for nid in self.neurons:
                self.neurons[nid].mutate(self.neurons)

    def applyState(self, state):
        for i in range(self.world_size):
            self.neurons[i].output = state[i]

    def thinkOnce(self, backward=True):
        population = [i for i in self.neurons.keys()]
        # forward
        random.shuffle(population)
        for p in population:
            self.neurons[p].doForward(self.neurons)
        # backward
        if backward:
            random.shuffle(population)
            for p in population:
                self.neurons[p].doBackward(self.neurons)

    def dumpBrain(self, filename):
        lines = []
        nid_s = list(self.neurons.keys())
        nid_s.sort()
        for nid in nid_s:
            neuron = self.neurons[nid]
            neuron.dendrites.sort(key=lambda x: x.from_nid)
            n_output = ', '.join([F"{d.from_nid}:{d.yes}:{d.no}:{d.getWeight():.2f}" for d in neuron.dendrites])
            lines.append(F"[{nid},{'R' if neuron.is_real else 'H'}]-> {n_output}")
        output = '\n'.join(lines)
        with open(filename, 'w') as fp:
            fp.write(output)

    def loadBrain(self, filename):
        lines = []
        with open(filename, 'r') as fp:
            lines = fp.readlines()
        self.neurons = {}
        for line in lines:
            left, right = line.split("->")
            nid, is_real = left.split(",")
            nid = int(nid.replace('[', ""))
            is_real = True if is_real[0] == 'R' else False
            dendrites = right.split(',')
            dendrites = [[int(x) for x in d.split(':')[:-1]] for d in dendrites]
            dendrites = [Dendrite(d[0], [d[1], d[2]]) for d in dendrites]
            self.neurons[nid] = Neuron(nid, is_real, dendrites)

    def sleep(self):
        print("wow")
