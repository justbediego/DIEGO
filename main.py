from world import World
from brain import Brain
import networkx as nx
import matplotlib.pyplot as plt

world = World(3)
brain = Brain(world.getSize())


def drawBrain():
    def getRGBAColor(weight):
        mag = abs(weight)
        return [
            0 if weight > 0 else 1,
            0,
            1 if weight > 0 else 0,
            1 if mag > 1 else 0 if mag < 0 else mag
        ]

    G = nx.Graph()
    node_colors = []
    for nid in brain.neurons:
        neuron = brain.neurons[nid]
        forward = 1 if neuron.forward > 1 else 0 if neuron.forward < 0 else neuron.forward
        G.add_node(nid)
        node_colors.append([0, forward, 0, 1])

    edge_colors = []
    for nid in brain.neurons:
        for d in brain.neurons[nid].dendrites:
            G.add_edge(nid, d.from_nid)
            edge_colors.append(getRGBAColor(d.weight))

    plt.clf()
    nx.draw(
        G,
        node_color=node_colors,
        with_labels=True,
        font_color='white',
        pos=nx.circular_layout(G),
        edge_color=edge_colors
    )
    plt.pause(0.05)


if __name__ == '__main__':
    while True:
        world.newState()
        brain.applyState(world.getState())
        for i in range(10):
            brain.thinkOnce()
            drawBrain()
