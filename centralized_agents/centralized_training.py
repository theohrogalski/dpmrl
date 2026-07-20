import torch
import sys
sys.path.append("/home/rogal/dpmrl/custom_environment/centralized_agents/env/centralized_graph_env.py")
from env import centralized_graph_env
from random import randint
from torch.distributions import Categorical
import argparse 
import networkx
import time
import pathlib
import logging
import math
import os
import matplotlib
from matplotlib import pyplot as plt
from torch.nn.functional import mse_loss
import random
from model_variants.centralized_full_model import centralized_full_model

class dpmrl_trainer:

    def __init__(self,model,max_iters:int, seed:int, saving_dir:pathlib.Path,training_id:str=str(random.randint(1,1000)),max_moves:int=1):
        self.seed = seed
        self.training_id = training_id
        self.max_iters = max_iters
        self.random_num = randint(1,9999999999)
        self.saving_dir:pathlib.Path = saving_dir
        self.max_moves = max_moves
        self.model = model
        print(f"self model is {self.model}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        #self.device = "cpu"
    def save_marl_checkpoint(self,episode, obs_net, unc_net,model, optimizer,epoch, training_id:str,ag_num=0, num_nodes=0,random_seed=0):
        """_save_marl_checkpoint saves a richly named .pt file containing a trained model at a certain point, including all relevant weights: each of the _

        Args:
            episode (_int_): _The current episode as an integer_
            obs_nets : _Observation processing network_
            unc_nets : _Uncertainty networks used for estimating the uncertainty in the system_
            model (_str_): _The current model being used to solve the problem_
            optimizers : _The optimizer objects_
            epoch (_int_): _The number of the current epoch_
            path (str, optional): _The path to where the checkpoint should be saved_. Defaults to "./checkpoints/".
            ag_num (int, optional): _The index of the current agent_. Defaults to 0.
            num_nodes (int, optional): __. Defaults to 0.
            random_seed (int, optional): _The random seed_. Defaults to 0.
        """

        checkpoint = {
            'episode': episode,
            'obs_state_dict': obs_net.state_dict(),
            'unc_state_dict': unc_net.state_dict(),
            'opt_state_dict': optimizer.state_dict() ,
        }
       # print(self.training_seed)
        if not os.path.exists(f"./checkpoints/training_id_{self.training_id}/seed_{self.seed}"):
            os.makedirs(f"./checkpoints/training_id_{self.training_id}/seed_{self.seed}")
        torch.save(checkpoint, f"""./checkpoints/training_id_{self.training_id}/seed_{self.seed}/checkpoint_model_{model.__name__}_ep_{episode}_seed_{random_seed}_trainingid_{training_id}.pt""")
            

    cur_length_list = []
    def diagnostic_plots(self,step,reward_history,epoch,uncertainty_history,neural_net_history):
        """
        Saves a figure to the /results folder.
        """
        
        matplotlib.use('Agg') 
        
        fig,ax= plt.subplots(3,3)
        ax1,ax2,ax3,ax4,ax5, ax6, ax7, ax8, ax9 = ax.flatten()
        agent_reward_list = [ax2, ax3, ax4, ax5]
        net_loss_list = [ax6, ax7, ax8, ax9]

        ax1.set_title("Uncertainty History")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Uncertainty")
        ax1.plot(uncertainty_history)   
        
        i=0 

        for axes in agent_reward_list:
            
            axes.set_xlabel("Timesteps")
            axes.set_ylabel("Reward per agent")
            axes.set_title(f"Reward History: agent_{i}")
            print(type(reward_history[f"agent_{i}"]))
            axes.plot(reward_history[f"agent_{i}"])
            i+=1


        plt.savefig(f"step:_{step}_epoch:_{epoch}_final_5.png")
        plt.close()


    def compute_ac_loss(self,log_prob, value, reward, next_value, done, gamma=0.99):
        """
        _This functions calculates the combined reward for the actor-critic 
        setup via the difference between the value and the reward combined with the next value_

        """
        mask = 1 - int(done)
        target = reward + (gamma * next_value * mask)
        advantage = target - value
        actor_loss = - log_prob * advantage
        critic_loss = mse_loss(value, target)
        total_loss = actor_loss + (0.5 * critic_loss)
        return total_loss
   
    def train_loop(self,num_nodes,num_agents,random_seed):
        torch.autograd.set_detect_anomaly(True)
        average_uncertainty_over_time = 0
        # Configuring the logger
        if not os.path.exists(f"./logs/training_id_{self.training_id}/seed_{self.seed}"):
            os.makedirs(f"./logs/training_id_{self.training_id}/seed_{self.seed}")
        logging.basicConfig(filename=f"./logs/training_id_{self.training_id}/seed_{self.seed}/{self.model.__name__}_{num_nodes}_{num_agents}_{random_seed}.log", level=logging.INFO)
        logging.info("started logging...")
        env = centralized_graph_env.CentralizedGraphEnv(num_nodes=num_nodes,num_agents=num_agents,seed=random_seed,render_mode="human",max_moves=self.max_moves,graph_selection=0)
        # Single observation processing network 
        self.obs_net = self.model(env.graph.number_of_nodes())
        self.obs_net = self.obs_net.to(self.device)
        x_state = None
        last_state = None
        optimizers = torch.optim.Adam(self.obs_net.parameters())
        gamma = 0.99
        critic_loss_dict:dict = {}
        reward_total = 0
        reward_history:dict = {agent:[] for agent in env.possible_agents}

        critic_loss_dict = {}
        self.net_loss:list = [] 
        max_iters=self.max_iters
        episode_num=0
        uncertainty_history:list = []
        start_time = time.time()
        while env.agents and max_iters>episode_num:
            
            if env.num_moves % env.max_moves == 0 and env.num_moves !=0:
                print("saving checkpoint")
                average_uncertainty_over_time = ((average_uncertainty_over_time*(episode_num+1)) + env.tot_unc)/((episode_num+1)*env.max_moves)
                logging.info(f"Average uncertainty at episode {episode_num}: {average_uncertainty_over_time}")

                print(f"Time elapsed for episode {episode_num}: {time.time()-start_time} seconds, {(time.time()-start_time)/60} mins")
                start_time = time.time()
                self.save_marl_checkpoint(episode = episode_num, obs_net = self.obs_net, model=self.model,unc_net=env.neural_model, training_id=self.training_id,optimizer=optimizers,epoch=episode_num,ag_num=num_agents,num_nodes=num_nodes,random_seed=random_seed)
                
                #self.diagnostic_plots(step=env.num_moves, reward_history=reward_history,epoch=env.num_epochs,uncertainty_history=uncertainty_history,neural_net_history=self.net_loss)
                
                env.reset()                
                uncertainty_history = []

                episode_num+=1
                logging.info(f"Iteration Changing to {episode_num}")

            total_loss = 0
            actions={}

            log_prob={}
            values={}

            
            for agent in env.agents:
                #DEBUG 
                #print(env.agent_position)
                #print(env.action_mask_to_node)
                #print(f"type of agent is {env.agent_position[agent]}")

                # /DEBUG
            
                if x_state is not None:
                    last_state = x_state
                logits, value, x_state, edges = self.obs_net(mental_map_nx=env.mental_map, mask=env.action_mask_to_node[int(agent[6:])],unc_net=env.neural_model,num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]],position=env.agent_position[agent])
                if last_state is None:
                    last_state = x_state
                #print(f" x_state shape here is {x_state.shape}")
                #assert x_state.shape == torch.Size([env.num_nodes,3])
                #assert last_state.shape == torch.Size([e3])
                unc_loss = env.neural_model.update_estimator(x = x_state.detach(),last_x=last_state, edge_index = edges,move_num=env.num_moves)
                #print(f"unc loss is {unc_loss}")
                #print(f"logit shape is {logits.shape}")
                #assert logits.shape == torch.Size([50])
                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()

                #print(f"action shape is {actions[agent].shape}")
                #DEBUG
                #print(f"Actions for {agent} are {actions[agent]}")
                #/DEBUG
                values[agent]=value

            obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent in env.agents:
               # print(env.agents)
                reward = torch.tensor([rewards[agent]], device=self.device)

                _,next_val,_,_ = self.obs_net(env.mental_map, env.action_mask_to_node[int(agent[6:])],env.neural_model,num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]], position =env.agent_position[agent])
                if env.step==env.max_moves-1:
                    done=1
                else:
                    done=0

                    log_prob = dist.probs[actions[agent]].item()
                    total_loss += self.compute_ac_loss(log_prob, value, reward, next_val, done)
                    logging.info(f"{env.num_moves}, {agent}, {reward.item():.2e} ,{int(total_loss.item())}, {actions[agent].item()}, {env.tot_unc}, {int(value.item())}, {int(next_val.item())}, {env.occupied_targets}, {unc_loss}, {env.longest_time_without_a_visit}")
                    reward_history[agent].append(rewards)
            optimizers.zero_grad()
            total_loss.backward()
            optimizers.step()
            self.net_loss.append(float(env.neural_model.loss_data[-1]))
            uncertainty_history.append(env.tot_unc)


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Trainer",
                                    description="""This module trains a team of agents with the given model,
                                    use build-on modules (e.x. ablation.py to auto-train)
                                    """)
    parser.add_argument("--training_id",type=str)

    parser.add_argument("--seed",type=int)
    parser.add_argument("--num_agents",type=int)
    parser.add_argument("--num_nodes",type=int)

    parser.add_argument("--model",type=str,default="centralized_full_model")
    parser.add_argument("--max_iters",type=int)
    parser.add_argument("--max_moves",type=int)

    # Saving directory
    saving_dir = pathlib.Path.cwd()/ "checkpoints"

    args = parser.parse_args()
    #print(args.model)
    model_string:str = args.model

    model = globals().get(model_string)
    #print(model)
    train_obj=dpmrl_trainer(max_iters=args.max_iters,model = model, training_id=args.training_id, max_moves=args.max_moves, saving_dir=saving_dir)
    
    nodes_for_data=args.num_nodes
    
    num_agents_for_testing=args.num_agents
    
    train_obj.train_loop(num_nodes = args.num_nodes, num_agents = args.num_agents, random_seed = args.seed)