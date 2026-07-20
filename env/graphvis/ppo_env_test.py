import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import time
import graphviz
import matplotlib.pyplot as plt
from stable_baselines3 import DQN,PPO, A2C
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

from custom_environment.env.graphvis.cycle_graph_env import GraphEnv

env=GraphEnv(num_nodes=20)

env = wrappers.OrderEnforcingWrapper(env)

parallel_env = aec_to_parallel(env)

vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
vec_env = ss.concat_vec_envs_v1(vec_env, 4, num_cpus=1, base_class="stable_baselines3")
vec_env = VecMonitor(vec_env,filename="./log_dir")
model = model_ppo = DQN("MlpPolicy", vec_env, verbose=1)
