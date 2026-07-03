import torch
from custom_environment.decentralized_agents.decentralized_graph_env import GraphEnv
import random
import ast
import argparse
from matplotlib import pyplot as plt
from custom_environment.decentralized_agents.model_variants.models_no_collision import models_no_collision
from custom_environment.decentralized_agents.model_variants.models_extra_attention_d import models_extra_attention
from custom_environment.decentralized_agents.model_variants.models_no_dcbf_d import models_no_dcbf
from custom_environment.decentralized_agents.model_variants.models_no_dcbf_no_state_estimation_d import models_no_dcbsf_no_state_est
from custom_environment.decentralized_agents.model_variants.models_full_model_d import models_full_model
from custom_environment.decentralized_agents.model_variants.models_no_state_estimation import models_no_state_est
import logging
import networkx as nx
from time import time
import os

import itertools

from datetime import datetime

from torch.distributions import Categorical

from custom_environment.training.neural_model import uncertainty_estimator as ue 

class algorithm_evaluator():
    def __init__(self):
        
        self.rand=random.randint(0,99999)
        #self.model_list = [models_no_dcbsf_no_state_est, models_extra_attention, models_full_model]
        
        self.device= "cuda"
    def sit_on_nodes(self):
        logging.basicConfig(filename=f'sitonnodes.log', level=logging.INFO)
        logger = logging.getLogger(f"sitonnodes")
        nodes_for_data=[50]
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
                        print(total_uncertainty_ever)
                        uncertainty_history=[]
                        num_iters+=1
                        print(f"Num iters for sit {num_iters+1}/100")
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
                logger.info(env.longest_time_without_a_visit)
                logger.info(time()-time_start)



    def full_model(self,num_nodes,num_agents,ckpt,model,log_name, max_iters,seed=0):
        log_dir = "logs/evaluation"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_filename = f"{log_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),  
                 ]
            )

        logging.info(f"Beginning eval. loop for seed {seed},max. iters:{max_iters}, num nodes: {num_nodes}, num agents: {num_agents}")    
        logging.info("num_moves, agent, reward, action, uncertainty, occ_nodes")

        env = GraphEnv(num_nodes=num_nodes,num_agents=num_agents,max_moves=500)
        "Evaluation for a pre-determined checkpoint for the full algorithm. Uses the self.ckpt as the path for loading models."
        check_dict  = torch.load(f"./checkpoints/{ckpt}.pt")

        obs_nets = check_dict["obs_state_dict"]

        unc_nets = check_dict["unc_state_dict"]
        
        

        uncertainty_history = []
        num_iters=0
        
        print(f"model is {model}")
        print(f"checkpoint is {ckpt}")


        total_uncertainty_ever = 0
        obs_net:dict = {agent:model(env.graph.number_of_nodes()) for agent in env.possible_agents}
        for agent,model in obs_net.items():
            model.load_state_dict(obs_nets[agent])
            model.to(self.device)
        agent_to_net:dict = {agent:ue(5,out_dim=1,hidden_dim=5,num_nodes=num_nodes,agent_name=agent) for agent in env.possible_agents}
        for agent, model in agent_to_net.items():
            
            model.load_state_dict(unc_nets[agent])
            model.to(self.device)
        actions={}
        while env.agents and num_iters<max_iters:
            time_start=time()
            if env.num_moves%500==0 and env.num_moves!=0:
                env.reset()
                total_uncertainty_ever+=sum(uncertainty_history)
                print(total_uncertainty_ever)
                uncertainty_history=[]
                num_iters+=1
                print(f"num iters: {num_iters}")
                
            for agent in env.agents:
                logits, value, x_state, edges = obs_net[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],agent_to_net[agent], num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]],position=env.agent_position[agent])
               
               

                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()
               
                logging.info(f"{env.num_moves}, {agent}, {env.rewards[agent]}  {actions[agent].item()}, {env.tot_unc}, {env.occupied_targets}")

            _,_,_,_,_ = env.step(actions)
            uncertainty_history.append(env.tot_unc)
        logging.info(f"{num_agents}_{num_nodes}_{log_name}_{max_iters}_iters")
        logging.info(total_uncertainty_ever)
        
        logging.info(env.longest_time_without_a_visit)
        logging.info(time()-time_start)               
        print(total_uncertainty_ever)
        return total_uncertainty_ever
    def partial_model(self,data):
        statistics=[]
    
    def random(self):
        logger = logging.getLogger(f"log_random")    
        logging.basicConfig(filename=f'log_random.log', level=logging.INFO)

        nodes_for_data=[20,50,100]
        num_agents_for_testing=[1,2,4,15]
        
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
                        print(total_uncertainty_ever)
                        
                        
                        env.reset()
                        uncertainty_history=[]
                        num_iters+=1
                        print(f"Num iters for Random at {num_iters}/100")
                    for agent in env.agents:
                        actions[agent]=torch.tensor(random.sample(self.get_legal_actions(env.agent_position[agent],env.action_mask_to_node),1))
                    _, _, _,_, _ = env.step(actions)
                    uncertainty_history.append(env.tot_unc)
                logger.info(f"{agents_num}_{nodes_num}")
                logger.info(total_uncertainty_ever)
                logger.info(env.longest_time_without_a_visit)
                logger.info(time()-time_start)
    def grazing(self):
        logger = logging.getLogger(f"log_grazing") 
        starting_time=datetime.now().strftime('%H:%M:%S')   
        logging.basicConfig(filename=f'log_grazing_{starting_time}.log', level=logging.INFO)
        logging.info(f"Starting Logging at {starting_time}")
        nodes_for_data=[50]
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
                        print(total_uncertainty_ever)
                        
                        
                        env.reset()
                        uncertainty_history=[]
                        num_iters+=1
                        print(f"Num iters for Random at {num_iters}/100")
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
            print(f"nn is {nn}")
            print(f"ag is {ag}")
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
    
    empty_ckpt_list=[]
    for ckpt in args.ckpt_list:
        for rs in args.random_seed_list:            
            empty_ckpt_list.append(ckpt+rs)
    ckpt_list = empty_ckpt_list

    eval=algorithm_evaluator()
    
    

    for ckpt in empty_ckpt_list:
        print(ckpt)
        seed=ckpt[-4:-1]
        for model in args.model_list :
            model = ast.literal_eval(model)
            print(ckpt)
            if model.__name__ in ckpt:
                print(f"{model.__name__} --- {ckpt}")
                model_class=model
        
        
        eval.full_model(num_nodes=args.num_nodes,num_agents=args.num_agents,ckpt=ckpt,model=model_class,log_name=f"{model.__name__}_data_collection",max_iters=args.max_iters,seed=seed)
