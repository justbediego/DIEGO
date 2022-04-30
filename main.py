from world import World
from brain import Brain
import networkx as nx
import matplotlib.pyplot as plt

world = World(1)
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

    G = nx.DiGraph()
    for nid in brain.neurons:
        neuron = brain.neurons[nid]
        output = 1 if neuron.output > 1 else 0 if neuron.output < 0 else neuron.output
        G.add_node(nid, color=[0, output, 0, 1])

    for nid in brain.neurons:
        for d in brain.neurons[nid].dendrites:
            G.add_edge(d.from_nid, nid, color=getRGBAColor(d.getWeight()))

    plt.clf()
    edge_colors = nx.get_edge_attributes(G, 'color').values()
    node_colors = list(nx.get_node_attributes(G, 'color').values())
    pos = nx.circular_layout(G)
    nx.draw(
        G,
        node_color=node_colors,
        with_labels=True,
        font_color='white',
        pos=pos,
        edge_color=edge_colors
    )
    plt.pause(0.01)


if __name__ == '__main__':
    while True:
        world.newState()
        brain.applyState(world.getState())
        for i in range(10):
            brain.thinkOnce()
        drawBrain()
