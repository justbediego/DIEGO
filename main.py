from world import World

world3 = World(3)

for i in range(10):
    world3.newState()
    print(world3.getState())