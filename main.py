from world import World
from brain import Brain

world = World(3)
brain = Brain(world.getSize())

while True:
    world.newState()
    brain.applyState(world.getState())
    for i in range(100):
        brain.thinkOnce()