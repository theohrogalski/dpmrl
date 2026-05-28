import torch
from decentralized_graph_env import GraphEnv
import random
from matplotlib import pyplot as plt
from models_no_collision import models_no_collision
from models_extra_attention_d import models_extra_attention
from models_no_dcbf_d import models_no_dcbf
from models_no_dcbf_no_state_estimation_d import models_no_dcbsf_no_state_est
from models_full_model_d import models_full_model
from models_no_state_estimation import models_no_state_est
import logging
import networkx as nx
from time import time
import os
#agents, optimizers = torch.load("")
import itertools
#test_env = GraphEnv()
from datetime import datetime

from torch.distributions import Categorical

from neural_model import uncertainty_estimator as ue 

class algorithm_evaluator():
    def __init__(self):
        #self.ckpt= ckpt
        self.rand=random.randint(0,99999)
        self.model_list = [models_extra_attention,models_no_dcbf,models_no_dcbsf_no_state_est,models_full_model,models_no_state_est,models_no_collision]
        
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
        
        # 3. Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),  # Writes to the subfolder
                 ]
            )

        logging.info(f"Beginning eval. loop for seed {seed},max. iters:{max_iters}, num nodes: {num_nodes}, num agents: {num_agents}")    
        logging.info("num_moves, agent, reward, action, uncertainty, occ_nodes")

        env = GraphEnv(num_nodes=num_nodes,num_agents=num_agents,max_moves=500)
        "Evaluation for a pre-determined checkpoint for the full algorithm. Uses the self.ckpt as the path for loading models."
        check_dict  = torch.load(f"./checkpoints/{ckpt}.pt")

        obs_nets = check_dict["obs_state_dict"]

        unc_nets = check_dict["unc_state_dict"]
        
        #opt = check_dict["opt_state_dict"]

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
                #print(f"Num iters for sit {num_iters}/20")
            for agent in env.agents:
                logits, value, x_state, edges = obs_net[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],agent_to_net[agent], num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]],position=env.agent_position[agent])
               # print(f"logits for agent {agent} are {logits}")
               # print(f"pos for agent {agent} are {env.agent_position[agent]}")

                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()
               # print(env.rewards)
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
                        #plt.plot(uncertainty_history)
                        total_uncertainty_ever+=sum(uncertainty_history)
                        print(total_uncertainty_ever)
                        #plt.set_title("uncertainty_history_random")
                        #plt.show()
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
                        #plt.plot(uncertainty_history)
                        total_uncertainty_ever+=sum(uncertainty_history)
                        print(total_uncertainty_ever)
                        #plt.set_title("uncertainty_history_random")
                        #plt.show()
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
                          #  print(f"{agent} is going to {actions[agent]}")

            
                        elif agent_path_dict[agent] != []:
                            actions[agent]=torch.tensor(agent_path_dict[agent].pop(0))
                           # print(f"{agent} is going to {actions[agent]}")
                        else:
                            actions[agent]=torch.tensor(env.agent_position[agent])
                           # print(f"{agent} is staying")
                        



                    _, _, _,_, _ = env.step(actions)
                    uncertainty_history.append(env.tot_unc)
                logger.info(f"{agents_num}_{nodes_num}")
                logger.info(total_uncertainty_ever)
                logger.info(env.longest_time_without_a_visit)
                logger.info(time()-time_start)
#eto : eval type object
def automatic_evaluation_for_grazing(eto):
    #eto.random()

    #eto.sit_on_nodes()
    #eto.grazing()
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
    checkpoint_list_no_=["_ckpoint_500_50_4_99_<class 'models_extra_attention_d.models_extra_attention'>_","_ckpoint_500_50_4_99_<class 'models_no_dcbf_d.models_no_dcbf'>_","_ckpoint_500_50_4_99_<class 'models_full_model_d.models_full_model'>_","_ckpoint_500_50_4_99_<class 'models_no_state_estimation.models_no_state_est'>_","_ckpoint_500_50_4_99_<class 'models_no_dcbsf_no_state_estimation_d.models_no_dcbf_no_state_est'>_"]
    
    second_run = ["_ckpoint_500_50_4_99_<class 'models_no_dcbsf_no_state_estimation_d.models_no_dcbf_no_state_est'>_"]
    evals=[]
    no_collision = ["ckpoint_dense_500_final_50_4_100_<class 'models_full_model_d.models_full_model'>_"]
    random_seed_list = ["103"]
    
    
    #ckpt_list=["saving ckpt ./checkpoints/_ckpoint_500_50_4_99_<class 'models_full_model_d.models_full_model'>_422","saving ckpt ./checkpoints/_ckpoint_500_50_4_99_<class 'models_full_model_d.models_full_model'>_"]
    
    third=["_ckpoint_dense_500_final_50_4_99_<class 'models_full_model_d.models_full_model'>_"]
    ckpt_list=[]
    model_list = [models_no_dcbf,models_no_dcbsf_no_state_est,models_full_model,models_no_state_est,models_no_collision]
    for ckpt in third:
        for rs in random_seed_list:            
            ckpt_list.append(ckpt+rs)
    #print(ckpt_list)

    eval=algorithm_evaluator()
    # eval.grazing()
    #eval.sit_on_nodes()

    for ckpt in ckpt_list:
        print(ckpt)
        seed=ckpt[-4:-1]
        for model in eval.model_list:
            if model.__name__ in ckpt:
                print(f"{model.__name__} --- {ckpt}")
                model_class=model
        
   
        eval.full_model(num_nodes=50,num_agents=4,ckpt=ckpt,model=model_class,log_name=f"Final_Evals",max_iters=20,seed=seed)
