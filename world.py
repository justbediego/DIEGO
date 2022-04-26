import random


class World:
    def __init__(self, order):
        self.order = order
        self.state = [0 for _ in range(order * 3 + 1)]

    def newState(self):
        self.state[0:(self.order * 2)] = [1 if random.random() > .5 else 0 for _ in range(self.order * 2)]
        next = 0
        for i in range(self.order):
            x = self.state[i]
            y = self.state[i + self.order]
            z = x + y + next
            self.state[i + self.order * 2] = 1 if (z == 1 or z == 3) else 0
            next = 1 if z > 1 else 0
        self.state[self.order * 3] = next

    def getState(self):
        return self.state
