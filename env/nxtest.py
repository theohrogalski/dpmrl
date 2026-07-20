import networkx as nx

G = nx.Graph(uncertainty=0)
G.graph
G.add_node(1, uncertainty=0)
G.add_nodes_from([3], time="2pm")
G.nodes[1]
{'time': '5pm'}
G.nodes[1]["room"] = 714  # node must exist already to use G.nodes
del G.nodes[1]["room"]  # remove attribute
print(list(G.nodes(data=True)))
