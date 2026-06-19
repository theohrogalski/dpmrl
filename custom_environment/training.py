from decentralized_graph_env import GraphEnv
import torch
import os
from models_full_model_d import models_full_model
from random import randint
from torch.distributions import Categorical
import argparse 
import ast
import os
import logging
import matplotlib
from matplotlib import pyplot as plt
from torch.nn.functional import mse_loss

class trainer:

    def __init__(self,model,max_iters:int,saving_dir:str,max_moves:int=1):

        self.max_iters=max_iters
        self.random_num=randint(1,9999999999)
        self.saving_dir=saving_dir
        self.max_moves=max_moves
        self.model = model
        
        
        
    def save_marl_checkpoint(self,episode, obs_nets, unc_nets,model, optimizers,epoch, path="./checkpoints/",ag_num=0, num_nodes=0,random_seed=0):
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
        # Create directory if it doesn't exist
        if not os.path.exists(path+self.saving_dir):
            os.makedirs(path+self.saving_dir)

        checkpoint = {
            'episode': episode,
            'obs_state_dict': {agent: net.state_dict() for agent, net in obs_nets.items()},
            'unc_state_dict': {agent: net.state_dict() for agent, net in unc_nets.items()},
            'opt_state_dict': {agent: opt.state_dict() for agent, opt in optimizers.items()},
        }

        temp_path = f"{path}ckpoint_dense_{episode}_final_{num_nodes}_{ag_num}_{epoch}_{self.model}_{random_seed}.tmp"
        final_path = f"{path}ckpoint_dense_{episode}_final_{num_nodes}_{ag_num}_{epoch}_{self.model}_{random_seed}.pt"
        print(f"saving ckpt {path}_ckpoint_{episode}_{num_nodes}_{ag_num}_{epoch}_{self.model}_{random_seed}")
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, final_path)
        
        torch.save(checkpoint, f"{path}latest.pt")
    

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
            axes.plot(reward_history)
            i+=1
    #--------------------------------------------------------------


        plt.savefig(f"step:_{step}_epoch:_{epoch}_final_5.png")
        plt.close()


    def compute_ac_loss(self,log_prob, value, reward, next_value, done, gamma=0.99):
        """
        This functions calculates the combined reward for the actor-critic 
        setup via the difference between the value and the reward combined with the next value.

        Args:
            log_prob (_type_): _description_
            value (_type_): _description_
            reward (_type_): _description_
            next_value (_type_): _description_
            done (function): _description_
            gamma (float, optional): _description_. Defaults to 0.99.

        Returns:
            _type_: _description_
        """
        mask = 1 - int(done)
        target = reward + (gamma * next_value * mask)
        advantage = target - value
        actor_loss = - log_prob * advantage
        critic_loss = mse_loss(torch.Tensor([value]), torch.Tensor([target]),reduction='mean')
        total_loss = actor_loss + (0.5 * critic_loss)
        assert total_loss.shape == torch.Size([1])
        return total_loss

   
    def train_loop(self,num_nodes,num_agents,random_seed):
        # Configuring the logger
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
        net_loss:dict = {agent:[] for agent in env.possible_agents}
        max_iters=self.max_iters
        num_iters=0
        uncertainty_history:list = []


        while env.agents and max_iters>num_iters:
            if env.num_moves%env.max_moves == 0 and env.num_moves !=0:
                self.save_marl_checkpoint(episode=env.num_moves,obs_nets=obs_nets, model=self.model,unc_nets=env.agent_to_net,optimizers=optimizers,epoch=env.num_epochs,ag_num=num_agents,num_nodes=num_nodes,random_seed=random_seed)
                self.diagnostic_plots(step=env.num_moves, reward_history=reward_history[agent],epoch=env.num_epochs,uncertainty_history=uncertainty_history,neural_net_history=net_loss[agent])
                env.reset()
                
                uncertainty_history = []
                reward_history:dict = {agent:[] for agent in env.possible_agents}
                num_iters+=1
                logger.info(f"Iteration Changing to {num_iters}")

            actions={}

            #step_data={}

            log_prob={}
            values={}
            unc_loss_dict = {}
            # Deprecated step_data approach

            #step_data={}
            
            for agent in env.agents:
                logits, value, x_state, edges = obs_nets[agent](mental_map_nx=env.mental_map[agent], mask=env.action_mask_to_node[int(agent[6:])],unc_net=env.agent_to_net[agent],num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]],position=env.agent_position[agent])
                unc_net = env.agent_to_net[agent]
                unc_loss = unc_net.update_estimator(x_state.detach(), edges,move_num=env.num_moves)
                unc_loss_dict[agent]=unc_loss

                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()
                               
                values[agent]=value
                """step_data[agent] = {
                    "log_prob":log_prob,
                    "value": value,
                    "prediction": unc_net(x_state, edges,env.num_moves).detach() 
                    }"""

            """for agent in env.agents:
                logger.info(f"Action dict is {torch.max(actions[agent])}")"""
            obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent in env.agents:
                reward = torch.tensor([rewards[agent]], device=dev)

                _,next_val,_,_ = obs_nets[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],env.agent_to_net[agent],num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]], position =env.agent_position[agent])
                value = values[agent]
                log_prob = actions[agent]
                if env.step==env.max_moves-1:
                    done=1
                else:
                    done=0
                total_loss = self.compute_ac_loss(log_prob[agent], value, reward, next_val, done)

                optimizers[agent].zero_grad()
                logger.info(f"{env.num_moves}, {agent}, {reward.item()} ,{int(total_loss.item())}, {actions[agent].item()}, {env.tot_unc}, {int(value.item())}, {int(next_val.item())}, {env.occupied_targets}, {int(unc_loss_dict[agent])}")
                total_loss.backward()
                optimizers[agent].step()
                
            # Logging
            for agent in env.agents:
                reward_history[agent].append(rewards[agent])
                net_loss[agent].append()
            uncertainty_history.append(env.tot_unc)


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Trainer",
                                    description="""This module trains a team of agents with the given model,
                                    use build-on modules (e.x. ablation.py to auto-train)
                                    """)
    parser.add_argument("--saving_dir",action='store')
    parser.add_argument("--seed_list",type=list)
    parser.add_argument("--num_agents_list",type=list)
    parser.add_argument("--num_nodes_list",type=list)

    parser.add_argument("--model",action='store')
    parser.add_argument("--max_iters",type=int)
    parser.add_argument("--max_moves",type=int)
    parser.add_argument("--max_iters")
    args = parser.parse_args()
    train_obj=trainer(max_iters=args.max_iters,model=ast.literal_eval(args.model), max_moves=args.max_moves,saving_dir=args.saving_dir)
    nodes_for_data=args.num_nodes_list
    num_agents_for_testing=args.num_agents_list
    for nn in nodes_for_data:
        for ag in num_agents_for_testing:
            for random_seed in args.seed_list:

                train_obj.train_loop(num_nodes=nn,num_agents=ag,random_seed=random_seed)