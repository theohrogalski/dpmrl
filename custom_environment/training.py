from custom_graph_env import GraphEnv
import torch
import os
from torch.distributions import Categorical
import argparse as argp
import logging
from math import trunc
import numpy as np
import os
import logging
from matplotlib import pyplot as plt
from torch.nn.functional import mse_loss
from gymnasium.wrappers import RecordEpisodeStatistics
class trainer():

    def __init__(self,model):
        self.model = model
        
    def save_marl_checkpoint(self,episode, obs_nets, unc_nets, optimizers,epoch, path="./checkpoints/",ag_num=0, n_num=0):
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Pack everything into one dictionary
        checkpoint = {
            'episode': episode,
            # Save every agent's model weights
            'obs_state_dict': {agent: net.state_dict() for agent, net in obs_nets.items()},
            'unc_state_dict': {agent: net.state_dict() for agent, net in unc_nets.items()},
            'opt_state_dict': {agent: opt.state_dict() for agent, opt in optimizers.items()},
        }

        # Save to a temporary file first, then rename (prevents corruption if job dies mid-save)
        temp_path = f"{path}_ckpt_{episode}_{n_num}_{ag_num}_{epoch}_{self.model}.tmp"
        final_path = f"{path}_ckpt_{episode}_{n_num}_{ag_num}_{epoch}_{self.model}.pt"
        print("saving ckpt")
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, final_path)
        
        # Also keep a 'latest' pointer for easy reloading
        torch.save(checkpoint, f"{path}latest.pt")
        #print(f"--- Checkpoint saved at Episode {episode} ---")
    logger = logging.getLogger("logger_train")
    logging.basicConfig(filename='debug_8.log', level=logging.INFO)
    #print("logger created")
    logger.info("------ Logger Started ------")
    logger.info("num_moves, agent, total_loss, action, uncertainty, value, next_val, occ_nodes, unc_loss")

    cur_length_list = []
    def save_diagnostic_plots(self,step, agents,reward_history,epoch,uncertainty_history):
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
        # 1. Calculate Target (TD Target)
        # If done, next value is 0

        mask = 1 - int(done)
        
        target = reward + (gamma * next_value * mask)
        
        # 2. Calculate Advantage (Target - Baseline)
        ##print(f"vale {value}")

        # Detach target because we don't want to backprop through the target for the critic
        ##print(f"target {target}")

        advantage = target - value
        ##print(f"adv {advantage}")
        # 3. Actor Loss: -log_prob * advantage
        ##print(f"log {log_prob}")
        #print(log_prob)
        actor_loss = - log_prob * advantage
        

        # 4. Critic Loss: MSE(value, target)
        # Use SmoothL1 or MSE
        


        critic_loss = mse_loss(torch.Tensor([value]), torch.Tensor([target]),reduction='mean')
        ##print((f"act {actor_loss}"))
        ##print((f"crt {critic_loss}"))
        # 5. Total Loss
        total_loss = actor_loss + (0.5 * critic_loss)
        ##print(f"tot {total_loss}")
        assert total_loss.shape == torch.Size([1])
        return total_loss

    def train_loop(self,num_nodes,num_agents):
        logger = logging.getLogger(f"fm_{num_nodes}_{num_agents}")
        logging.basicConfig(filename=f"fm_{num_nodes}_{num_agents}", level=logging.INFO)

        env = GraphEnv(num_nodes=num_nodes,num_agents=num_agents)
        ##print(f" here3 {env.graph.nodes()}")
        ##print(env.agent_position
        if torch.cuda.is_available():
            dev="cuda"
        else:
            dev="cpu"
        obs_nets:dict = {agent:self.model(env.graph.number_of_nodes())for agent in env.possible_agents}
        for nets in obs_nets.values():
            # #print("a")
            nets.to(dev)
        optimizers = {agent:torch.optim.Adam(obs_nets[agent].parameters()) for agent in env.agents}

        gamma = 0.99

        critic_loss_dict:dict = {}
        ##print("starting")
        import logging
        reward_total = 0

        # Hyperparameters
        GAMMA = 0.99
        critic_loss_dict = {}
        # Main Episode Loop
        reward_history:dict = {agent:[] for agent in env.possible_agents}
        max_iters=25
        num_iters=0
        uncertainty_history:list = []
        while env.agents and max_iters>num_iters:
            #print(env.num_moves)
            if env.num_moves%env.max_moves == 0 and env.num_moves !=0:
                self.save_marl_checkpoint(episode=env.num_moves,obs_nets=obs_nets,unc_nets=env.agent_to_net,optimizers=optimizers,epoch=env.num_epochs,ag_num=num_agents,n_num=num_nodes)
                #self.save_diagnostic_plots(step=env.num_moves,agents=env.possible_agents,reward_history=reward_history[agent],epoch=env.num_epochs,uncertainty_history=uncertainty_history)
                env.reset()
                uncertainty_history = []
                reward_history:dict = {agent:[] for agent in env.possible_agents}
                num_iters+=1


            actions={}
            step_data={}
            log_prob={}
            unc_loss_dict = {}
            # --- PHASE 1: COLLECT ACTIONS ---
            for agent in env.agents:
            # 1. Run the forward pass
                #print(f"agnet pos is {env.agent_position[agent]}")
                logits, value, x_state, edges = obs_nets[agent](mental_map_nx=env.mental_map[agent], mask=env.action_mask_to_node[int(agent[6:])],unc_net=env.agent_to_net[agent],num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]],position=env.agent_position[agent])
                unc_net = env.agent_to_net[agent]
                # This call now only handles the GCN logic
                unc_loss = unc_net.update_estimator(x_state.detach(), edges,move_num=env.num_moves)
                #logger.info(f"unc_loss @ {env.num_moves} is {unc_loss}")
                #print(f"logits are {logits}")
                #print(logits.shape)
                unc_loss_dict[agent]=unc_loss

                dist = Categorical(logits=logits)
                actions[agent] = dist.sample()
                
                log_prob[agent] = (actions[agent])
                # Store for Phase 3
                step_data[agent] = {
                    "log_prob":log_prob,
                    "value": value,
                    "prediction": unc_net(x_state, edges,env.num_moves).detach() # For Task 2 (DCBF)
                    }

            # --- PHASE 2: STEP ENVIRONMENT ---
            """for agent in env.agents:
                logger.info(f"Action dict is {torch.max(actions[agent])}")"""
            ##print(actions)
            obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent, data in step_data.items():
            # 1. Prepare Ground Truths (Moved to GPU)
                reward = torch.tensor([rewards[agent]], device=dev)

                _,next_val,_,_ = obs_nets[agent](env.mental_map[agent], env.action_mask_to_node[int(agent[6:])],env.agent_to_net[agent],num_moves=env.num_moves,neighbors=env.action_mask_to_node[env.agent_position[agent]], position =env.agent_position[agent])
                # 2. Retrieve the stored Log Prob and Value from Phase 1
                #log_prob = data["log_prob"] #
                value = data["value"]
                #print(value)
                log_prob = data["log_prob"]      
                if env.step==env.max_moves-1:
                    done=1
                else:
                    done=0
                
                # 3. CALCULATE THE COMBINED LOSS
                # This function (compute_ac_loss) combines Actor and Critic math
                #   log_prob=torch.max(log_prob)
                total_loss = self.compute_ac_loss(log_prob[agent], value, reward, next_val, done)

                # 4. PERFORM THE UPDATE
                # This updates BOTH the Actor and Critic weights simultaneously
                optimizers[agent].zero_grad()
                logger.info(f"{env.num_moves}, {agent}, {int(total_loss.item())}, {actions[agent].item()}, {env.tot_unc}, {int(value.item())}, {int(next_val.item())}, {env.occupied_targets}, {int(unc_loss_dict[agent])}")
                total_loss.backward()
                optimizers[agent].step()
                
            # Logging
            for agent in env.agents:
                reward_history[agent].append(rewards[agent])
            uncertainty_history.append(env.tot_unc)


if __name__=="__main__":
   parser = argp.ArgumentParser(description="Parser for training loop")