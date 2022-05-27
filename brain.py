import random
import numpy as np


def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


def activation(x):
    return sigmoid(x)
    # return x if x > 0 else 0


def dActivation(x):
    tmp = sigmoid(x)
    return tmp * (1 - tmp)
    # return 1 if x > 0 else 0


class Dendrite:
    def __init__(self, from_nid, weight=None, age=1):
        self.from_nid = from_nid
        self.age = age
        if weight is not None:
            self.weight = weight
        else:
            self.weight = 0
            while self.weight == 0:
                self.weight = random.random()
            self.weight = (self.weight * 2 - 1) * .1

    def getWeight(self):
        return self.weight

    def decreaseWeight(self, amount):
        learning_rate = .2 * (0.99995 ** self.age)
        self.weight = self.weight - (amount * learning_rate)
        self.age = self.age + 1


class Neuron:
    def __init__(self, nid, is_real=False, age=1, bias=0):
        self.nid = nid
        self.is_real = is_real
        self.dendrites = []
        self.age = age
        self.output = 0
        self.backward = None
        self.incoming_energy = 0
        self.bias = Dendrite(-1, bias, age)

    def getScore(self):
        if len(self.dendrites) < 1:
            return 0
        return np.sum([abs(d.getWeight()) for d in self.dendrites])
        # return np.log(self.age) * best

    def mutate(self, neurons):
        all_dendrites = [
            i for i in neurons.keys()
            if (not self.is_real and i < self.nid) or (self.is_real and i != self.nid)
        ]
        current_dendrites = [d.from_nid for d in self.dendrites]
        possible = [i for i in all_dendrites if i not in current_dendrites]
        random.shuffle(possible)
        min_new = min(2, len(possible))
        max_new = min(3, len(possible))
        new_dendrite_count = random.randint(min_new, max_new)
        # new_dendrite_count = dendrite_count - len(self.dendrites)
        for i in range(new_dendrite_count):
            self.dendrites.append(Dendrite(possible[i]))

    def doForward(self, neurons):
        z = self.bias.getWeight()
        for d in self.dendrites:
            other = neurons[d.from_nid]
            z = z + other.output * d.getWeight()
        self.incoming_energy = z
        o = activation(z)
        if self.is_real:
            self.backward = o - self.output
        else:
            self.output = o

    def doBackward(self, neurons):
        if self.backward is not None:
            # print(self.backward)
            diff = self.backward * dActivation(self.incoming_energy)
            for d in self.dendrites:
                other = neurons[d.from_nid]
                if not other.is_real:
                    other.backward = (0 if other.backward is None else other.backward) + diff * d.getWeight()
                # update
                d.decreaseWeight(diff * other.output)
            self.age = self.age + 1
            self.bias.decreaseWeight(diff)
            self.backward = None


class Brain:
    def __init__(self, world_size, file_name=None, max_generation=30, min_generation=5):
        self.world_size = world_size
        self.max_generation = max_generation
        self.min_generation = min_generation
        if file_name is not None:
            self.loadBrain(file_name)
        else:
            self.neurons = {i: Neuron(i, True) for i in range(world_size)}
            for i in range(world_size, world_size + max_generation):
                self.neurons[i] = Neuron(i)
            for nid in self.neurons:
                self.neurons[nid].mutate(self.neurons)

    def dumpBrain(self, filename):
        lines = []
        nid_s = list(self.neurons.keys())
        nid_s.sort()
        for nid in nid_s:
            neuron = self.neurons[nid]
            neuron.dendrites.sort(key=lambda x: x.from_nid)
            n_output = ', '.join([F"{int(d.from_nid)}:{d.weight:.2f}:{int(d.age)}" for d in neuron.dendrites])
            lines.append(F"[{nid},{'R' if neuron.is_real else 'H'},{int(neuron.age)},{neuron.bias.weight:.2f}]-> {n_output}")
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
            nid, is_real, age, bias = left.replace('[', "").replace(']', "").split(",")
            nid = int(nid)
            is_real = True if is_real == 'R' else False
            age = int(age)
            bias = float(bias)
            self.neurons[nid] = Neuron(nid, is_real, age, bias)
            if ':' in right:
                dendrites = right.split(',')
                dendrites = [d.split(':') for d in dendrites]
                dendrites = [Dendrite(int(d[0]), float(d[1]), int(d[2])) for d in dendrites]
                self.neurons[nid].dendrites = dendrites

    def applyState(self, state):
        for i in range(self.world_size):
            self.neurons[i].output = state[i]

    def thinkOnce(self, backward=True):
        population = [i for i in self.neurons.keys() if not self.neurons[i].is_real]
        population.extend([i for i in self.neurons.keys() if self.neurons[i].is_real])
        # forward
        for p in population:
            self.neurons[p].doForward(self.neurons)
        if backward:
            # backward
            population.reverse()
            for p in population:
                self.neurons[p].doBackward(self.neurons)

    def sleep(self):
        n_map = [[nid, self.neurons[nid].getScore()] for nid in self.neurons if not self.neurons[nid].is_real]
        n_map = [[x[0], -np.inf if (x[1] == 0 or np.isnan(x[1])) else x[1]] for x in n_map]
        n_map.sort(key=lambda x: x[1], reverse=True)
        deleting_nid_s = [x[0] for x in n_map[self.min_generation:]]
        for nid in deleting_nid_s:
            self.neurons.pop(nid)
        for nid in self.neurons:
            neuron = self.neurons[nid]
            for i in range(len(neuron.dendrites) - 1, -1, -1):
                dendrite = neuron.dendrites[i]
                if dendrite.from_nid in deleting_nid_s:
                    neuron.dendrites.pop(i)
        # # generating new
        # n_count = len(n_map) - len(deleting_nid_s)
        # last_nid = np.max([x[0] for x in n_map]) + 1
        # for i in range(self.max_generation - n_count):
        #     self.neurons[i + last_nid] = Neuron(i + last_nid)
        #     self.neurons[i + last_nid].mutate(self.neurons)
        # for i in range(self.world_size):
        #     self.neurons[i].mutate(self.neurons)
