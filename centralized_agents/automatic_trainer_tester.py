# Outline: Create a loop for testing with a number of seeds, agents, 
# Need: a loop that trains, then tests, a specific model with an easily locatable checkpoint file
# Structure: Either train all at once then check then test then check OR train once, check, test once, check etc
import centralized_training
from model_variants import centralized_full_model

from eval import algorithm_evaluator

import tqdm



import pathlib
class auto_trainer_tester:
    def __init__(self):
        self.num_nodes_to_agents:dict = {100:4,200:4,50:4}

        self.random_seeds:list = [63458]
        
        self.checkpoint_list:list = []
        
        self.model_list:list = [centralized_full_model.centralized_full_model] 


        self.training_id = "solo3"
        path1 = pathlib.Path(f"./checkpoints/training_id_{self.training_id}/seed_63458/checkpoint_model_centralized_full_model_ep_99_seed_63458_trainingid_solo3_200.pt")
        path2 = pathlib.Path(f"./checkpoints/training_id_{self.training_id}/seed_63458/checkpoint_model_centralized_full_model_ep_99_seed_63458_trainingid_solo3_100.pt")
        path3 = pathlib.Path(f"./checkpoints/training_id_{self.training_id}/seed_63458/checkpoint_model_centralized_full_model_ep_99_seed_63458_trainingid_solo3_50.pt")
        self.checkpoint_dir = [path1, path2, path3]

        self.max_moves = 500

        self.max_iters = 100
        self.num_agents = 4

    def train(self,model, max_iters, max_moves, num_nodes, num_agents, seed):
        trainer = centralized_training.dpmrl_trainer(model = model, max_iters = max_iters,seed=seed, saving_dir = "./checkpoints",max_moves=max_moves, training_id = self.training_id)
        trainer.train_loop(num_nodes = num_nodes, num_agents = num_agents, random_seed = seed)

    def eval(self, num_nodes,num_agents, checkpoint:str,model,max_iters, seed,max_moves):
        eval = algorithm_evaluator(num_moves = max_moves, seed = seed, num_agents=(num_agents), num_nodes = num_nodes)
        eval.full_model(num_nodes=num_nodes,num_agents=num_agents,ckpt = checkpoint, model=model,log_name=f"decentralized_data_collection_{model.__name__}_{seed}_{num_agents}",max_iters=max_iters,seed = seed)
    def full_loop(self):
        """skip_seed_818 = 1
        for model_obj in tqdm.tqdm(self.model_list):
            print(f"setting model object to {model_obj}")
            for num_nodes, num_agents in self.num_nodes_to_agents.items():
                print(f"setting num nodes {num_nodes}, agents to {num_agents}")
                for seed in tqdm.tqdm(self.random_seeds):
                    print(f"Setting seed to {seed}")
                    print("Starting training...")
                    #self.train(model = model_obj, max_moves = self.max_moves, seed=seed max_iters = self.max_iters, num_nodes = num_nodes, num_agents = num_agents, seed = seed)
                    print("Starting evaluation...")
                    self.eval(model = model_obj, num_nodes = num_nodes, max_iters = self.max_iters, seed = seed, num_agents = num_agents, max_moves = self.max_moves, checkpoint = f"./checkpoints/training_id_{self.training_id}/seed_{self.seed}_checkpoint_model_{model_obj.__name__}_ep_{self.max_iters-1}_seed_{seed}_trainingid_{self.training_id-1}.pt")
            """
        new_checkpoint_ls:list = []
        for checkpoint in self.checkpoint_dir:
            #print(checkpoint)
            new_ckpt:list = [checkpoint]
            for seed in self.random_seeds:
                if str(seed) in str(checkpoint):
                    new_ckpt.append(seed)
            for num_nodes in self.num_nodes_to_agents.keys():
                if str(num_nodes) in str(checkpoint):
                    new_ckpt.append(num_nodes)
            for model_obj in self.model_list:
                if str(model_obj.__name__) in str(checkpoint):
                    new_ckpt.append(str(model_obj.__name__))
            
            #print(checkpoint)
            #print(new_ckpt)
            new_checkpoint_ls.append(new_ckpt)
        print(new_checkpoint_ls[0])
            
            
        #print(f"Evaluating {str(checkpoint)}")
        for full_checkpoint in new_checkpoint_ls :
    
          #  print(f"{full_checkpoint[0]} {full_checkpoint[1]} {full_checkpoint[2]} {full_checkpoint[3]}")
            print(getattr(globals().get(full_checkpoint[3]),str(full_checkpoint[3])))
            self.eval(model = getattr(globals().get(full_checkpoint[3]),str(full_checkpoint[3])), num_nodes = full_checkpoint[2], num_agents = self.num_agents, max_iters = 20, seed = full_checkpoint[1], max_moves = self.max_moves, checkpoint = full_checkpoint[0])


if __name__ == "__main__":
    automaticTrainerTester = auto_trainer_tester()
    automaticTrainerTester.full_loop()