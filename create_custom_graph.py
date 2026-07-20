import networkx as nx
import random 
import itertools
from random import randint
def create_custom_nx_graph(num_nodes=50, output_name:str="node_50", random_chance_param_target:int=20, random_chance_param_edge:int=20) -> None:
    r_graph = nx.Graph()
    for i in range(num_nodes):
        r_graph.add_node(int(i))

        r_graph.nodes[i]["uncertainty"] = 0
        r_graph.nodes[i]["agent_presence"] = 0
        r_graph.nodes[i]["target"] = 0
        print("got here")
        if random.randint(1,100)<random_chance_param_target:
            r_graph.nodes[i]["target"] = 1
    combinations_list = itertools.combinations(r_graph.nodes,2)
    for combo in combinations_list:
        if random.randint(1,100)<random_chance_param_edge:
            r_graph.add_edge(combo[0],combo[1])
    nx.write_graphml(r_graph,f"./graphs/{output_name}.graphml")
    ##print("got here")

if __name__=="__main__":
    create_custom_nx_graph(num_nodes=100,output_name="100_nodes",random_chance_param_edge=20,random_chance_param_target=16)
