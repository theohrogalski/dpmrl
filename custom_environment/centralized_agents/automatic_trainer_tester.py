# Outline: Create a loop for testing with a number of seeds, agents, 
# Need: a loop that trains, then tests, a specific model with an easily locatable checkpoint file
# Structure: Either train all at once then check then test then check OR train once, check, test once, check etc
from training import centralized_training
from model_variants import centralized_full_model

from eval.eval import algorithm_evaluator

import tqdm



import pathlib
class auto_trainer_tester:
    def __init__(self):
        self.num_nodes_to_agents:dict = {100:8,200:16,50:4}

        self.random_seeds:list = [818,312,10492]
        
        self.checkpoint_list:list = []
        
        self.model_list:list = [centralized_full_model.centralized_full_model] 


        self.training_id = 0

        self.max_moves=1_000

        self.max_iters = 100

    def train(self,model, max_iters, max_moves, num_nodes, num_agents, seed):
        trainer = centralized_training.dpmrl_trainer(model = model, max_iters = max_iters, saving_dir = "./checkpoints",max_moves=max_moves, training_id = self.training_id)
        self.training_id+=1
        trainer.train_loop(num_nodes = num_nodes, num_agents = num_agents, random_seed = seed)

    def eval(self, num_nodes,num_agents, checkpoint:str,model,max_iters, seed,max_moves):
        eval = algorithm_evaluator(num_moves = max_moves, seed = seed)
        eval.full_model(num_nodes=num_nodes,num_agents=num_agents,ckpt = checkpoint, model=model,log_name=f"centralized_data_collection_{model.__name__}",max_iters=max_iters,seed = seed)
    def full_loop(self):
        # Combinatorially tractable
        for model_obj in tqdm.tqdm(self.model_list):
            print(f"setting model object to {model_obj}")
            for num_nodes, num_agents in self.num_nodes_to_agents.items():
                print(f"setting num nodes {num_nodes}, agents to {num_agents}")
                for seed in tqdm.tqdm(self.random_seeds):
                    print(f"Setting seed to {seed}")
                    print("Starting training...")
                    self.train(model = model_obj, max_moves = self.max_moves, max_iters = self.max_iters, num_nodes = num_nodes, num_agents = num_agents, seed = seed)
                    print("Starting evaluation...")
                    self.eval(model = model_obj, num_nodes = num_nodes, max_iters = self.max_iters, seed = seed, num_agents = num_agents, max_moves = self.max_moves, checkpoint = f"./checkpoints/checkpoint_model_{model_obj.__name__}_ep_{self.max_iters-1}_seed_{seed}_trainingid_{self.training_id-1}.pt")
                

if __name__ == "__main__":
    automaticTrainerTester = auto_trainer_tester()
    automaticTrainerTester.full_loop()