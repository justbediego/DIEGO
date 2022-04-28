import random


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
        self.forward = 0
        self.backward = 0

    def mutate(self, neurons):
        population = [i for i in neurons.keys() if i != self.nid]
        random.shuffle(population)
        self.dendrites = [Dendrite(population[i]) for i in range(random.randint(2, 3))]

    def doForward(self, neurons):
        forward = 0
        for d in self.dendrites:
            other = neurons[d.from_nid]
            forward = forward + (other.forward * d.weight)
        # activation (relu)
        forward = forward if forward > 0 else 0
        if self.is_real:
            self.backward = self.forward - forward
        else:
            self.forward = forward

    def doBackward(self, neurons):
        if self.backward == 0:
            return
        learning_rate = .1
        for d in self.dendrites:
            other = neurons[d.from_nid]
            if other.forward > 0:
                charge = self.backward * d.weight
                d.weight = d.weight + (charge * learning_rate)
                if not other.is_real:
                    other.backward = charge


class Brain:
    def __init__(self, world_size):
        self.world_size = world_size
        self.neurons = {i: Neuron(i, True) for i in range(world_size)}
        for i in range(world_size, 30):
            self.neurons[i] = Neuron(i)
        for nid in self.neurons:
            self.neurons[nid].mutate(self.neurons)

    def applyState(self, state):
        for i in range(self.world_size):
            self.neurons[i].forward = state[i]

    def thinkOnce(self):
        population = [i for i in self.neurons.keys()]
        # forward
        random.shuffle(population)
        for p in population:
            self.neurons[p].doForward(self.neurons)

        # backward
        random.shuffle(population)
        for p in population:
            self.neurons[p].doBackward(self.neurons)
