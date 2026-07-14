import networkx as nx
import random
import itertools

def create_custom_nx_graph(num_nodes:int, output_name:str="random_output_graph", random_chance_param_target:int=20, random_chance_param_edge:int=20) -> None:
    r_graph = nx.Graph()
    for i in range(num_nodes):
        r_graph.add_node(int(i))

        r_graph.nodes[i]["uncertainty"] = 0
        r_graph.nodes[i]["agent_presence"] = 0
        r_graph.nodes[i]["target"] = 0
        
        if random.randint(1,100)<random_chance_param_target:
            r_graph.nodes[i]["target"] = 1
    combinations_list = itertools.combinations(r_graph.nodes,2)
    for combo in combinations_list:
        if random.randint(1,100)<random_chance_param_edge:
            r_graph.add_edge(combo[0],combo[1])
    nx.write_graphml(r_graph,f"./{output_name}.graphml")

if __name__ == "__main__":
    create_custom_nx_graph(output_name = "200_nodes_",num_nodes = 200, random_chance_param_target = 8)
    
    create_custom_nx_graph(output_name = "100_nodes",num_nodes = 200, random_chance_param_target = 4)
