import torch
import networkx as nx
import pettingzoo
from gymnasium import spaces
from pettingzoo import AECEnv

class AEC_Graph(AECEnv) :

    metadata = { "name": "aec_pmg","render_modes": ["human"],}

    def __init__(self, graph_data, n_agents, max_steps=100, num_targets=6, view_size=2, seed=None, grid_size=(10,10),render_mode=None) :
        self.graph = nx.Graph(uncertainty=0)
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        max_neighbors=0
        for nodes in self.graph:
            curr_neighbors = len(nx.neighbors(nodes))
            if curr_neighbors > max_neighbors:
                max_neighbors = curr_neighbors

        self.action_space = {agent : (spaces.Discrete(50, start = 0),spaces.Discrete(max_neighbors+1))
                             for agent in self.possible_agents
                             } #Velocity, Move to neighbor (discrete decision)
        self.agent_positions = self.graph
        self.observation_space = {agent : spaces.Dict({
            "current_node":self.agent_position[agent],
            "surrounding_nodes":self
            })
                                for agent in self.possible_agents}
    def reset(self):

    def observe():
    def choose_neighbor(self,agent):
     
    def step(self,action):
        if action[1] == 0:
            pass
        else for action 

    def render():
