import custom_environment.env.rayrlib_env as rayrlib_env
import networkx as nx
import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class GraphEnv(MultiAgentEnv):

    def __init__(self ,n_agents:int,grid_size:int,config=None):
        super().__init__()
        self.graph = nx.Graph()
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents 
        nodelist = [(i, {"uncertainty":0}) for i in range(start=1,stop=10)]
        self.graph.add_nodes_from(nodelist)
        for nodes in self.graph:
            curr_neighbors = len(nx.neighbors(nodes))
            if curr_neighbors > max_neighbors:
                max_neighbors = curr_neighbors
        self.transit = {agent:False
                        for agent in self.possible_agents
                        }
        self.observation_spaces = {
            agent : gym.spaces.Graph()
            for agent in self.possible_agents
         }
        self.action_spaces = {
            agent : gym.spaces.Discrete(5)
            for agent in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        
        # return observation dict and infos dict.
        return {
                "agent_1": np.array([0.0, 1.0, 0.0, 0.0]),
                "agent_2": np.array([0.0, 0.0, 1.0]),
            }, {} #info
    def step(self, action_dict):
        observations,infos,rewards,truncations,terminations = {}
        for agent in self.agents:
            if action_dict[agent] == 0 :
                pass
            if 0 < action_dict:
                target = self.graph()
            
    def compute_transit_time(length_of_edge, velocity):
        return length_of_edge / velocity
    def observe(self,agent):
        obsdict = {}
        for agent in self.agents:
            obsdict[agent] = 0
