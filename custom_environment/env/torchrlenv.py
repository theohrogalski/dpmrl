from torchrl.envs import EnvBase
import networkx as nx
from torchrl.data import CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec,Observation

import torch

class par_env(EnvBase) :
    def __init__(self, graph_preset, n_agents,device):
        self.graph = nx.Graph(graph_preset)
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.action_space = CompositeSpec(
            discrete = DiscreteTensorSpec(n=5), # which nodes to visit 
            continuous = BoundedTensorSpec(shape=torch.Size([2]),device='cuda', minimum=0, maximum=50) # this is the velocity, ie the energy
        )
        self.observation_spec = 
