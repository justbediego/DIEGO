import random
import numpy as np


def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


def activation(x):
    # return sigmoid(x)
    return x if x > 0 else 0


def dActivation(x):
    # tmp = sigmoid(x)
    # return tmp * (1 - tmp)
    return 1 if x > 0 else 0


class Dendrite:
    def __init__(self, from_nid, yes_no=None):
        self.from_nid = from_nid
        if yes_no is not None:
            self.yes, self.no = yes_no
        else:
            self.yes = 0
            while self.yes == 0:
                self.yes = random.random()
            self.no = self.yes
            while self.no == self.yes or self.no == 0:
                self.no = random.random()
            self.yes = self.yes + 10
            self.no = self.no + 10

    def getWeight(self):
        # 1 to -1
        return (self.yes - self.no) / (self.yes + self.no)
        # return self.weight

    def increaseWeight(self, amount):
        if amount > 0:
            self.yes = self.yes + amount
        elif amount < 0:
            self.no = self.no - amount


class Neuron:
    def __init__(self, nid, is_real=False, age=1):
        self.nid = nid
        self.is_real = is_real
        self.dendrites = []
        self.age = age

    def getScore(self):
        best = np.max([abs(d.getWeight()) for d in self.dendrites])
        return np.log(self.age) * best

    def mutate(self, neurons):
        current_from_s = [d.from_nid for d in self.dendrites]
        population = [i for i in neurons.keys() if i != self.nid and i not in current_from_s]
        random.shuffle(population)
        if len(population) == 0:
            return
        new_dendrite_count = random.randint(2, min(3, len(population)))
        for i in range(new_dendrite_count):
            self.dendrites.append(Dendrite(population[i]))

    def doForward(self, neurons):
        z = 0
        for d in self.dendrites:
            other = neurons[d.from_nid]
            z = z + other.output * d.getWeight()
        self.incoming_energy = z
        o = activation(z)
        if self.is_real:
            self.backward = self.output - o
        else:
            self.output = o

    def doBackward(self, neurons):
        if self.backward is not None:
            self.age = self.age + 1
            diff = self.backward * dActivation(self.incoming_energy)
            for d in self.dendrites:
                other = neurons[d.from_nid]
                if not other.is_real:
                    other.backward = (0 if other.backward is None else other.backward) + diff * d.getWeight()
                # update
                d.increaseWeight(diff * other.output)
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
            n_output = ', '.join(
                [F"{int(d.from_nid)}:{d.yes:.2f}:{d.no:.2f}:{d.getWeight():.2f}" for d in neuron.dendrites])
            lines.append(F"[{nid},{'R' if neuron.is_real else 'H'},{neuron.age}]-> {n_output}")
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
            nid, is_real, age = left.replace('[', "").replace(']', "").split(",")
            nid = int(nid)
            is_real = True if is_real == 'R' else False
            age = int(age)
            self.neurons[nid] = Neuron(nid, is_real, age)
            if ':' in right:
                dendrites = right.split(',')
                dendrites = [[float(x) for x in d.split(':')] for d in dendrites]
                dendrites = [Dendrite(d[0], [d[1], d[2]]) for d in dendrites]
                self.neurons[nid].dendrites = dendrites

    def applyState(self, state):
        for i in range(self.world_size):
            self.neurons[i].output = state[i]

    def thinkOnce(self, backward=True):
        # clearing
        for p in range(self.world_size, len(self.neurons)):
            self.neurons[p].output = 0
            self.neurons[p].backward = None
            self.neurons[p].incoming_energy = 0
        population = [i for i in self.neurons.keys()]
        # forward
        random.shuffle(population)
        for p in population:
            self.neurons[p].doForward(self.neurons)
        if backward:
            # backward
            random.shuffle(population)
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
        # generating new
        n_count = len(self.neurons)
        last_nid = max(self.neurons) + 1
        for i in range(self.max_generation - n_count):
            self.neurons[i + last_nid] = Neuron(i + last_nid)
        for nid in self.neurons:
            self.neurons[nid].mutate(self.neurons)
