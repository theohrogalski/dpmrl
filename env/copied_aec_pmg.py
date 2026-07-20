from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces
import numpy as np
import random

class GraphEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "graph_env_v0"}

    def __init__(self, graph: nx.Graph):
        super().__init__()
        self.graph = graph
        self.possible_agents = ["agent_0"]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}
        self.current_node = 0
        self._max_neighbors = max(len(list(graph.neighbors(n))) for n in graph.nodes)

        self.action_spaces = {agent: spaces.Discrete(self._max_neighbors + 1) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict({
                "current_node": spaces.Discrete(len(graph.nodes)),
                "neighbors": spaces.MultiBinary(len(graph.nodes))
            }) for agent in self.agents
        }

    def observe(self, agent):
        neighbors = list(self.graph.neighbors(self.current_node))
        neighbor_vec = np.zeros(len(self.graph.nodes), dtype=np.int8)
        neighbor_vec[neighbors] = 1
        return {
            "current_node": self.current_node,
            "neighbors": neighbor_vec
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_node = 0
        self._agent_selector = iter(self.agents)
        self.agent_selection = next(self._agent_selector)
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        return self.observe(self.agent_selection)

    def step(self, action):
        if self.terminations[self.agent_selection]:
            self._was_dead_step(action)
            return

        neighbors = list(self.graph.neighbors(self.current_node))
        if action == 0:
            pass  # Stay
        elif 1 <= action <= len(neighbors):
            self.current_node = neighbors[action - 1]
        else:
            # Invalid action (e.g. move to non-existent neighbor)
            self.rewards[self.agent_selection] = -1
            self.terminations[self.agent_selection] = True
            return

        # Rewarding staying/moving — customizable
        self.rewards[self.agent_selection] = 1

        # Terminate condition (e.g., reach specific node or after steps)
        self.terminations[self.agent_selection] = False

        # Switch agent (only 1 here)
        self.agent_selection = next(self._agent_selector, self.agents[0])

    def render(self):
        print(f"Agent at node {self.current_node}")

    def close(self):
        pass
