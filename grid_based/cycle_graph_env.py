import asyncio
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import time
import signal
import graphviz
import random
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from pettingzoo.utils import aec_to_parallel
import supersuit as ss
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecMonitor
from pettingzoo.test import api_test
import pettingzoo
from stable_baselines3.common.evaluation import evaluate_policy
import functools
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
import time
#print(plt.get_backend())

class GraphEnv(pettingzoo.AECEnv):
    def __init__(self, num_nodes=10, num_agents=2, seed=1,render_mode="human"):
        self.seed=seed
        plt.subplot(2,1,1)
        plt.clf()
        plt.subplot(2,1,2)
        plt.clf()
        self.lx=[]
        self.render_mode = 2
        self.render_flag = False
        self.ly=[]
        self.metadata = {
        "render_modes": ["human"],   # or ["human"] if you want GUI rendering
        "name": "graph_env_v0",
        "is_parallelizable":True
        }
        self.render_mode=render_mode
        self.np_random_seed = int(np.random.randint(1, 10 + 1))
        self.graph = nx.cycle_graph(num_nodes)  # simple graph
        #print(self.graph)
        self.possible_agents = [f"agent_{k}" for k in range(num_agents)]
        self.total_map_observation = {agent:("") for agent in self.possible_agents}
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.num_nodes = num_nodes
        self.action_spaces = {agent:spaces.Discrete(num_nodes) for agent in self.possible_agents}  # move to node
        self.observation_spaces = {agent:gym.spaces.Discrete(num_nodes) for agent in self.possible_agents}
        self.agents = self.possible_agents
        self._cumulative_rewards = {agent:0 for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.num_moves = 0
        self.max_uncertainty:int = 100
        self.mistakes = {agent:0 for agent in self.possible_agents}
        self.node_unc = {node:0 for node in self.graph}
        #print(f"node uncertainty is {self.node_unc}")
        self.rewards = {agent:0 for agent in self.agents}
        self.infos = {agent:{} for agent in self.agents}
        self.covered = set()
        #self.per_agent_covered = {agent:set() for agent in self.possible_agents}
        self.mental_map= {agent:nx.Graph() for agent in self.possible_agents}
    def reset(self, options=None,seed=None):
        self.covered = set()
        self.num_moves=0
        self.node_unc = {node:0 for node in self.graph}
        for agent in self.possible_agents:
            self.state[agent] = random.randint(0,self.num_nodes-1)
            self.rewards[agent] = 0
            self._cumulative_rewards[agent] = 0
            self.lx = []
            self.ly = []
            plt.subplot(2,1,1)
            plt.clf()
            plt.subplot(2,1,2)
            plt.clf()

            self.infos[agent] = {}
        self.agents = (self.possible_agents).copy()
        self._agent_selector = agent_selector(self.agents)
        
        if self._agent_selector:
            self.agent_selection = self._agent_selector.next()
        self.terminations = {agent:False for agent in self.agents}
        self.observations = {agent: 0 for agent in self.agents}

        self.truncations = {agent:False for agent in self.agents}
        self.timestep = 0
        return self.state, {}
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # We can seed the action space to make the environment deterministic.
        return gym.spaces.Discrete(3, seed=self.np_random_seed)
    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent) :
        return gym.spaces.Discrete(self.num_nodes)

    def step(self, action):
        for node in self.graph:
            #print(node)
            if self.node_unc[node]>=self.max_uncertainty:
                continue
            #print(self.state)
            elif (np.int64(node) in self.state.values()):
                
                #print("found")
                if self.node_unc[node]>0:
                    self.node_unc[node] -= list(self.state.values()).count(np.int64(node))
                    #print("reduced uncertainty")
                if self.node_unc[node]<0:
                    self.node_unc[node] = 0
                
            else:
                self.node_unc[node] += 1
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        # Wrap around logic (ie first-> last, last->first)
        if(self.state[self.agent_selection]==0 and action==0):
            self.state[self.agent_selection] = self.num_nodes-1
        elif(self.state[self.agent_selection]==self.num_nodes-1 and action==2):
            self.state[self.agent_selection] = 0
        else:
            #print("took grow")
            self.state[self.agent_selection] += action-1

        #(self.covered).add(self.state[self.agent_selection])
        if self._agent_selector.is_last():
            self.num_moves += 1
            
            # print(f"num moves is {self.num_moves}")
           # print(sum(self.node_unc.values()))
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= 1000 for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = self.state[
                    self.agents[1 - self.agent_name_mapping[i]]]
                

            # reward giving
        
                #print(-sum(self.node_unc.values()))
                self.rewards[i] = int(-sum(self.node_unc.values())/(self.num_nodes))
                #print(self.rewards[i])
        else:
            self._clear_rewards()
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        
        self._accumulate_rewards()
        (self.lx).append(self.num_moves)
        (self.ly).append(self._cumulative_rewards["agent_0"])
        if self.render_mode == "human":
            self.render()
    def observe(self, agent):
    # simplest: every agent just sees the global state
        return np.array(self.observations[agent])
    def render(self, total_reward=None):
        #plt.clf()
        #print(self.state)
        plt.subplot(2,1,1)
        nx.draw_networkx(self.graph, with_labels=True,pos=nx.spring_layout(self.graph,seed=0),
                node_color=
                [(min((list(self.state.values()).count(i)*100)/self.num_agents,0.99),
                  min(1, self.node_unc[i]/self.max_uncertainty),
                  min(1, self.node_unc[i]/self.max_uncertainty)
                  ) for i in self.graph.nodes()]
                  
                  )
        
        #print(self.ly)
        if self.num_moves%50==0:
            plt.subplot(2,1,2)
            plt.plot(self.lx,self.ly)
            plt.pause(1)
        #mngr = plt.get_current_fig_manager()
        #mngr.window.wm_geometry((f"500x500+100+100"))        # TkAgg backend (most common)

        #plt.ion()
        # to put it into the upper left corner for example:
        #plt.pause(0.75)

        #plt.close()
        
        
        
    
        
env=GraphEnv(num_nodes = 40, num_agents=6)

env = wrappers.OrderEnforcingWrapper(env)        
parallel_env = aec_to_parallel(env)

vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=8, base_class="stable_baselines3")
vec_env = VecMonitor(vec_env,filename="./log_dir")
"""model = DQN("MlpPolicy", vec_env, verbose=1)
timestart = time.time()
model.learn(total_timesteps=10,progress_bar=True)
print(f"total time = {time.time()-timestart}")
model.save(f"policy_")"""
#mean_reward, std_reward = evaluate_policy(model=model, env=vec_env)
#print(f"mean reward for DQN is {mean_reward}, std_reward for DQN is {std_reward}")
model_dqn = DQN("MlpPolicy", vec_env, verbose=1)
#model_ppo = PPO("MlpPolicy", vec_env, verbose =1)
mean_reward_dqn,std_reward_dqn= evaluate_policy(model=model_dqn, env=vec_env)
print(f"mean reward without any learning{mean_reward_dqn}")
model_dqn.learn(total_timesteps=300,progress_bar=True)
#model_ppo.learn(total_timesteps=100, progress_bar=True)
#model_dqn.save("dqn_model_savefile")


#model_ppo.save("dqn_save")
mean_reward_dqn,std_reward_dqn= evaluate_policy(model=model_dqn, env=vec_env)
#mean_reward_ppo, std_reward_ppo = evaluate_policy(model=model_ppo,env=vec_env)
print(f"mean reward for dqn is {mean_reward_dqn}")
#print(f"mean reward for dqn is {mean_reward_ppo}")

print("With num training steps being 100")
model_dqn.load("./dqn_model_savefile.zip")
mean_reward_dqn,std_reward_dqn= evaluate_policy(model=model_dqn, env=vec_env)
print(mean_reward_dqn)