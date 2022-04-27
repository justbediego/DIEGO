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
        self.state = 0
        self.dendrites = []

    def mutate(self, neurons):
        population = [i for i in neurons.keys()]
        random.shuffle(population)
        self.dendrites = [Dendrite(population[i]) for i in range(random.randint(2, 3))]


class Brain:
    def __init__(self, world_size):
        self.world_size = world_size
        self.neurons = {i: Neuron(i, True) for i in range(world_size)}
        for i in range(world_size, 100):
            self.neurons[i] = Neuron(i)
        for nid in self.neurons:
            self.neurons[nid].mutate(self.neurons)
        a = 12

    def applyState(self, state):
        for i in range(self.world_size):
            self.neurons[i].state = state[i]

    def thinkOnce(self):
        print(self.world_size)
