from decentralized_graph_env import GraphEnv
import torch
import os
from models_full_model_d import models_full_model

from torch.nn.modules.container import ParameterList
from torch.distributions import Categorical
from models_no_collision import models_no_collision
import argparse as argp
from math import trunc
import numpy as np
import os
import logging
from matplotlib import pyplot as plt
from torch.nn.functional import mse_loss
from gymnasium.wrappers import RecordEpisodeStatistics
class trainer():
    def __init__(self,model,max_iters,max_moves=1):
        self.max_iters=max_iters
        self.max_moves=max_moves
        self.model = model
        
        
        
    def save_marl_checkpoint(self,episode, obs_nets, unc_nets, optimizers,epoch, path="./checkpoints/",ag_num=0, n_num=0,random_seed=0):
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        checkpoint = {
            'episode': episode,
            'obs_state_dict': {agent: net.state_dict() for agent, net in obs_nets.items()},
            'unc_state_dict': {agent: net.state_dict() for agent, net in unc_nets.items()},
            'opt_state_dict': {agent: opt.state_dict() for agent, opt in optimizers.items()},
        }

        temp_path = f"{path}ckpoint_dense_{episode}_final_{n_num}_{ag_num}_{epoch}_{self.model}_{random_seed}.tmp"
        final_path = f"{path}ckpoint_dense_{episode}_final_{n_num}_{ag_num}_{epoch}_{self.model}_{random_seed}.pt"
        print(f"saving ckpt {path}_ckpoint_{episode}_{n_num}_{ag_num}_{epoch}_{self.model}_{random_seed}")
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, final_path)
        
        torch.save(checkpoint, f"{path}latest.pt")
    

    cur_length_list = []
    def diagnostic_plots(self,step, agents,reward_history,epoch,uncertainty_history):
        """
        Saves a diagnostic figure to the /results folder.
        """
        
        import matplotlib
        matplotlib.use('Agg') 
        
        fig,ax= plt.subplots(3,3)
        ax1,ax2,ax3,ax4,ax5, ax6,ax7,ax8,ax9 = ax.flatten()
        obj_list = [ax2,ax3,ax4,ax5]

        ax1.set_title("Uncertainty History")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Uncertainty")
        ax1.plot(uncertainty_history)   
        
        i=0 
        for axes in obj_list:
            
            axes.set_xlabel("Timesteps")
            axes.set_ylabel("Reward per agent")
            axes.set_title(f"Reward History: agent_{i}")
            axes.plot(reward_history)
            i+=1
    #--------------------------------------------------------------


        plt.savefig(f"step:_{step}_epoch:_{epoch}_final_5.png")
        plt.close()


    def compute_ac_loss(self,log_prob, value, reward, next_value, done, gamma=0.99):
        mask = 1 - int(done)
        target = reward + (gamma * next_value * mask)
        advantage = target - value
        actor_loss = - log_prob * advantage
        critic_loss = mse_loss(torch.Tensor([value]), torch.Tensor([target]),reduction='mean')
        total_loss = actor_loss + (0.5 * critic_loss)
        assert total_loss.shape == torch.Size([1])
        return total_loss

   
    def train_loop(self,num_nodes,num_agents):
        logger = logging.getLogger(f"fm_{num_nodes}_{num_agents}")
        logging.basicConfig(filename=f"fm_{num_nodes}_{num_agents}", level=logging.INFO)

        env = GraphEnv(num_nodes=num_nodes,num_agents=num_agents)
        if torch.cuda.is_available():
            dev="cuda"
        else:
            dev="cpu"
        obs_nets:dict = {agent:self.model(env.graph.number_of_nodes())for agent in env.possible_agents}
        for nets in obs_nets.values():
            nets.to(dev)
        optimizers = {agent:torch.optim.Adam(obs_nets[agent].parameters()) for agent in env.agents}
        gamma = 0.99
        critic_loss_dict:dict = {}
        reward_total = 0
        critic_loss_dict = {}
        reward_history:dict = {agent:[] for agent in env.possible_agents}
        max_iters=self.max_iters
        num_iters=0
        uncertainty_history:list = []
        while env.agents and max_iters>num_iters:
            if env.num_moves%env.max_moves == 0 and env.num_moves !=0:
                self.save_marl_checkpoint(episode=env.num_moves,obs_nets=obs_nets,unc_nets=env.agent_to_net,optimizers=optimizers,epoch=env.num_epochs,ag_num=num_agents,n_num=num_nodes,random_seed=random_seed)
                #self.diagnostic_plots(step=env.num_moves,agents=env.possible_agents,reward_history=reward_history[agent],epoch=env.num_epochs,uncertainty_history=uncertainty_history)
                env.reset()
                
                uncertainty_history = []
                reward_history:dict = {agent:[] for agent in env.possible_agents}
                num_iters+=1
                logger.info(f"Iteration Changing to {num_iters}")

            actions={}
            step_data={}
            log_prob={}
            unc_loss_dict = {}
            for agent in env.agents:
                logits, value, x_state, edges = obs_nets[agent](mental_map_nx=env.mental_map[agent], mask=env.action_mask_to_node[int(agent[6:])],unc_net=env.agent_to_net[agent],num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]],position=env.agent_position[agent])
                unc_net = env.agent_to_net[agent]
                unc_loss = unc_net.update_estimator(x_state.detach(), edges,move_num=env.num_moves)
                unc_loss_dict[agent]=unc_loss

                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()
                
                log_prob[agent] = (actions[agent])
                step_data[agent] = {
                    "log_prob":log_prob,
                    "value": value,
                    "prediction": unc_net(x_state, edges,env.num_moves).detach() 
                    }

            """for agent in env.agents:
                logger.info(f"Action dict is {torch.max(actions[agent])}")"""
            obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent, data in step_data.items():
                reward = torch.tensor([rewards[agent]], device=dev)

                _,next_val,_,_ = obs_nets[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],env.agent_to_net[agent],num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]], position =env.agent_position[agent])
                value = data["value"]
                log_prob = data["log_prob"]      
                if env.step==env.max_moves-1:
                    done=1
                else:
                    done=0
                total_loss = self.compute_ac_loss(log_prob[agent], value, reward, next_val, done)

                optimizers[agent].zero_grad()
                logger.info(f"{env.num_moves}, {agent}, {reward} ,{int(total_loss.item())}, {actions[agent].item()}, {env.tot_unc}, {int(value.item())}, {int(next_val.item())}, {env.occupied_targets}, {int(unc_loss_dict[agent])}")
                total_loss.backward()
                optimizers[agent].step()
                
            # Logging
            for agent in env.agents:
                reward_history[agent].append(rewards[agent])
                
            uncertainty_history.append(env.tot_unc)


if __name__=="__main__":
    homunculus=trainer(max_iters=1000,model=models_full_model,max_moves=500)
    nodes_for_data=[50]
    num_agents_for_testing=[4]
    for nn in nodes_for_data:
        for ag in num_agents_for_testing:
            for random_seed in [103,878,422]:

                homunculus.train_loop(num_nodes=nn,num_agents=ag,random_seed=random_seed)
