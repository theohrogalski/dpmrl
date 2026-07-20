import networkx as nx 
import matplotlib.pyplot as plt
def draw_town_figure():
    graph = nx.Graph()
    for i in range(10):
        graph.add_node(f"town_{i}")
    for node in graph:
        for node_two in graph:
             if node_two is not node:
                graph.add_edge(node,node_two)
    nx.draw_networkx(graph, with_labels=True)
    plt.show()

if __name__ == "__main__":
    draw_town_figure()