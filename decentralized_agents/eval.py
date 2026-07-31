import torch
from env.decentralized_graph_env import GraphEnv
import random
import ast
import argparse
from matplotlib import pyplot as plt
from model_variants import full_model, extra_attention_model,  no_collision_model, no_dcbf_model, no_state_est_model, no_state_est_no_dcbf_model
import logging
import networkx as nx
from time import time
import os

import itertools

from datetime import datetime

from torch.distributions import Categorical

from env.decentralized_state_estimator import uncertainty_estimator as ue 

class algorithm_evaluator():
    def __init__(self,num_moves:int, num_agents:int, num_nodes:int, seed:int):

        self.num_moves = num_moves
        
        torch.manual_seed(seed) 

        self.num_agents = num_agents

        self.num_nodes = num_nodes       
        
        self.device= "cuda"

    def sit_on_nodes(self):
        if not os.path.exists(f"./logs/eval/sit_on_nodes"):
            os.makedirs(f"./logs/eval/sit_on_nodes")
        log_file = f"./logs/eval/sit_on_nodes/sit.log"

        logger = logging.getLogger(log_file)
        logger.handlers.clear()  # Prevent duplicate handlers if function is called multiple times
        logger.addHandler(logging.FileHandler(log_file))
        logger.setLevel(logging.INFO)
        logger.info(f"model is sit on nodes")
        nodes_for_data=[50,100,200]
        num_agents_for_testing=[4]
        for nodes_num in nodes_for_data:
            for agents_num in num_agents_for_testing:
                time_start=time()
                env=GraphEnv(num_nodes=nodes_num,num_agents=agents_num)
                action_freeze={agent:0 for agent in env.possible_agents}
                
                actions={}
                uncertainty_history = []
                max_iters=20
                num_iters=0

                total_uncertainty_ever=0
                while env.agents and num_iters<max_iters:
                    if env.num_moves%500==0 and env.num_moves!=0:
                        env.reset()
                        total_uncertainty_ever+=sum(uncertainty_history)
                        logging.info(total_uncertainty_ever)
                        uncertainty_history=[]
                        num_iters+=1
                        logging.info(f"Num iters for sit {num_iters+1}/100")
                    for agent in env.agents:
                        if action_freeze[agent]==0:
                            if env.graph.nodes[env.agent_position[agent]]["target"]==1 and env.graph.nodes[env.agent_position[agent]]["agent_presence"]==0:
                                action_freeze[agent]=1
                                actions[agent] = torch.tensor(env.agent_position[agent])
                            else:
                                actions[agent]=torch.tensor(random.sample(self.get_legal_actions(env.agent_position[agent],env.action_mask_to_node),1))            
                        
                        else:
                            actions[agent] = torch.tensor(env.agent_position[agent])
                    _, _, _,_, _ = env.step(actions)

                    uncertainty_history.append(env.tot_unc)
                logger.info(f"{agents_num}_{nodes_num}")
                logger.info(total_uncertainty_ever)
                print(total_uncertainty_ever)

                logger.info(env.longest_time_without_a_visit)
                print(env.longest_time_without_a_visit)

                logger.info(time()-time_start)
                print(time()-time_start)


    def full_model(self,num_nodes,num_agents,ckpt,model,log_name, max_iters,seed=0):
        if not os.path.exists(f"./logs/eval/seed_{seed}"):
            os.makedirs(f"./logs/eval/seed_{seed}")
        log_file = f"./logs/eval/seed_{seed}/{model.__name__}_{num_nodes}_{num_agents}_{seed}.log"

        logger = logging.getLogger(log_file)
        logger.handlers.clear()  # Prevent duplicate handlers if function is called multiple times
        logger.addHandler(logging.FileHandler(log_file))
        logger.setLevel(logging.INFO)
        logger.info(f"model is {model}")

        env = GraphEnv(num_nodes=num_nodes,num_agents=num_agents,max_moves=self.num_moves)
        check_dict  = torch.load(ckpt)

        obs_nets = check_dict["obs_state_dict"]

        unc_nets = check_dict["unc_state_dict"]
        
        

        uncertainty_history = []
        num_iters=0
        
        logger.info(f"checkpoint is {ckpt}")

        total_uncertainty_ever = 0
        obs_net:dict = {agent:model(env.graph.number_of_nodes()) for agent in env.possible_agents}
        for agent,model in obs_net.items():
            model.load_state_dict(obs_nets[agent])
            model.to(self.device)
        agent_to_net:dict = {agent:ue(5,out_dim=1,hidden_dim=5,num_nodes=num_nodes,agent_name=agent, num_agents=self.num_agents, max_moves=self.num_moves) for agent in env.possible_agents}
        for agent, model in agent_to_net.items():
            
            model.load_state_dict(unc_nets[agent])
            model.to(self.device)
        actions={}
        while env.agents and num_iters<max_iters:
            time_start=time()
            if env.num_moves%500==0 and env.num_moves!=0:
                env.reset()
                total_uncertainty_ever+=sum(uncertainty_history)
                logger.info(total_uncertainty_ever)
                uncertainty_history=[]
                num_iters+=1
                logger.info(f"num iters: {num_iters}")
                
            for agent in env.agents:
                logits, value, x_state, edges = obs_net[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],agent_to_net[agent], num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]],position=env.agent_position[agent])
               
               

                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()
               
              #  logging.info(f"{env.num_moves}, {agent}, {env.rewards[agent]}  {actions[agent].item()}, {env.tot_unc}, {env.occupied_targets}")

            _,_,_,_,_ = env.step(actions)
            uncertainty_history.append(env.tot_unc)
        logger.info(f"{num_agents}_{num_nodes}_{log_name}_{max_iters}_iters")
        print(f"{num_agents}_{num_nodes}_{log_name}_{max_iters}_iters")
        logger.info(total_uncertainty_ever)
        print(total_uncertainty_ever)
        logger.info(env.longest_time_without_a_visit)
        print(env.longest_time_without_a_visit)
        logger.info(time()-time_start)             
        print(time()-time_start)  
        logger.info(total_uncertainty_ever)
        print(total_uncertainty_ever)
        return total_uncertainty_ever
    def partial_model(self,data):
        statistics=[]
    
    def random(self):
        if not os.path.exists(f"./logs/eval/random"):
            os.makedirs(f"./logs/eval/random")
        log_file = f"./logs/eval/random/random.log"

        logger = logging.getLogger(log_file)
        logger.handlers.clear()  # Prevent duplicate handlers if function is called multiple times
        logger.addHandler(logging.FileHandler(log_file))
        logger.setLevel(logging.INFO)
        logger.info(f"model is random")
        nodes_for_data=[50,100,200]
        num_agents_for_testing=[4]
        
        for nodes_num in nodes_for_data:
            for agents_num in num_agents_for_testing:
                time_start=time()
                env=GraphEnv(num_nodes=nodes_num,num_agents=agents_num)
                actions={}
                uncertainty_history = []
                max_iters = 20
                num_iters=0
                total_uncertainty_ever=0
                while env.agents and num_iters<max_iters:
                    if env.num_moves%500==0 and env.num_moves!=0:
                        
                        total_uncertainty_ever+=sum(uncertainty_history)
                        logging.info(total_uncertainty_ever)
                        
                        
                        env.reset()
                        uncertainty_history=[]
                        num_iters+=1
                        logging.info(f"Num iters for Random at {num_iters}/100")
                    for agent in env.agents:
                        actions[agent]=torch.tensor(random.sample(self.get_legal_actions(env.agent_position[agent],env.action_mask_to_node),1))
                    _, _, _,_, _ = env.step(actions)
                    uncertainty_history.append(env.tot_unc)
                logger.info(f"{agents_num}_{nodes_num}")
                logger.info(total_uncertainty_ever)
                logger.info(env.longest_time_without_a_visit)
                logger.info(time()-time_start)
    def grazing(self):
        if not os.path.exists(f"./logs/eval/grazing"):
                    os.makedirs(f"./logs/eval/grazing")
        log_file = f"./logs/eval/grazing/grazing.log"

        logger = logging.getLogger(log_file)
        logger.handlers.clear()  # Prevent duplicate handlers if function is called multiple times
        logger.addHandler(logging.FileHandler(log_file))
        logger.setLevel(logging.INFO)
        logger.info(f"model is grazing")
        nodes_for_data=[50,100,200]
        num_agents_for_testing=[4]
        
        for nodes_num in nodes_for_data:
            for agents_num in num_agents_for_testing:
                time_start=time()
                env=GraphEnv(num_nodes=nodes_num,num_agents=agents_num)
                agent_path_dict = {agent:[] for agent in env.possible_agents}
                actions={}
                uncertainty_history = []
                max_iters = 20
                num_iters=0
                total_uncertainty_ever=0
                while env.agents and num_iters<max_iters:
                    if env.num_moves%500==0 and env.num_moves!=0:
                        
                        total_uncertainty_ever+=sum(uncertainty_history)
                        logging.info(total_uncertainty_ever)
                        
                        
                        env.reset()
                        uncertainty_history=[]
                        num_iters+=1
                        logging.info(f"Num iters for grazing at {num_iters}/100")
                    for agent in env.agents:
                        if env.graph.nodes[env.agent_position[agent]]["uncertainty"]==0 and agent_path_dict[agent]==[]:
                            
                            max_unc=0
                            max_unc_node=env.agent_position[agent]
                            for node in env.graph.nodes():
                                for list in agent_path_dict.values():
                                    if list!=[]:
                                            if node==list[-1]:
                                                node=env.agent_position[agent]
                                if env.graph.nodes[node]["agent_presence"]==0:
                                    if env.graph.nodes[node]["uncertainty"]>max_unc:
                                        max_unc=env.graph.nodes[node]["uncertainty"]
                                        max_unc_node=node

                            agent_path_dict[agent]=nx.shortest_path(env.graph,env.agent_position[agent],target=max_unc_node)
                            actions[agent]=torch.tensor(agent_path_dict[agent].pop(0))
                          

            
                        elif agent_path_dict[agent] != []:
                            actions[agent]=torch.tensor(agent_path_dict[agent].pop(0))
                           
                        else:
                            actions[agent]=torch.tensor(env.agent_position[agent])
                           
                        



                    _, _, _,_, _ = env.step(actions)
                    uncertainty_history.append(env.tot_unc)
                logger.info(f"{agents_num}_{nodes_num}")
                logger.info(total_uncertainty_ever)
                logger.info(env.longest_time_without_a_visit)
                logger.info(time()-time_start)

def automatic_evaluation_for_grazing(eto):
    

    
    
    nodes_for_data=[50]
    num_agents_for_testing=[4]
    for nn in nodes_for_data:
        for ag in num_agents_for_testing:
            logging.info(f"nn is {nn}")
            logging.info(f"ag is {ag}")
            eto.grazing(num_agents=ag,num_nodes=nn)

"""if __name__=="__main__":
    eval=algorithm_evaluator()
    eval.grazing()"""
if __name__=="__main__":
    evals=[]
    
    parser = argparse.ArgumentParser(prog="Trainer",
                                    description="""This module trains a team of agents with the given model,
                                    use build-on modules (e.x. ablation.py to auto-train)
                                    """)
    parser.add_argument("--saving_dir",action='store')
    parser.add_argument("--ckpt_list",type=list)
    parser.add_argument("--random_seed_list",type=list)
    parser.add_argument("--num_nodes_list",type=list)
    #a list of strings that turns into an object via ast.literal_eval()
    parser.add_argument("--model_list",action='store',type=list)
    parser.add_argument("--num_nodes",type=int)
    parser.add_argument("--num_agents",type=int)
    parser.add_argument("--max_iters")
    args = parser.parse_args()
    eval = algorithm_evaluator(num_agents=4, num_moves=500, num_nodes = 10, seed=0)
    """  empty_ckpt_list=[]
    for ckpt in args.ckpt_list:
        for rs in args.random_seed_list:            
            empty_ckpt_list.append(ckpt+rs)
    ckpt_list = empty_ckpt_list

    eval=algorithm_evaluator()"""
    
    

    """for ckpt in empty_ckpt_list:
        logging.info(ckpt)
        seed=ckpt[-4:-1]
        for model in args.model_list :
            model = ast.literal_eval(model)
            logging.info(ckpt)
            if model.__name__ in ckpt:
                logging.info(f"{model.__name__} --- {ckpt}")
                model_class=model
        
        
        eval.full_model(num_nodes=args.num_nodes,num_agents=args.num_agents,ckpt=ckpt,model=model_class,log_name=f"{model.__name__}_data_collection",max_iters=args.max_iters,seed=seed)
    """
    eval.random()
    eval.grazing()
    eval.sit_on_nodes()