# Outline: Create a loop for testing with a number of seeds, agents, 
# Need: a loop that trains, then tests, a specific model with an easily locatable checkpoint file
# Structure: Either train all at once then check then test then check OR train once, check, test once, check etc
from model_variants import full_model, extra_attention_model,  no_collision_model, no_dcbf_model, no_state_est_model, no_state_est_no_dcbf_model
from model_variants import decentralized_training
from eval import algorithm_evaluator
import pathlib
import tqdm
import pathlib



class auto_trainer_tester:
    def __init__(self):
        self.num_nodes_to_agents:dict = {100:4,200:4,50:4}

        self.random_seeds:list = [818,312,10492]
      
        self.model_list:list = [
                                full_model.full_model, no_state_est_model.no_state_est_model,
                                extra_attention_model.extra_attention_model,
                                no_collision_model.no_collision_model, no_dcbf_model.no_dcbf_model,
                                ] 

        self.training_id = "last"

        self.max_moves = 500

        self.max_iters = 100

        # Placeholder 
        self.num_agents = 4

        path = pathlib.Path(f"./checkpoints/training_id_{self.training_id}")

        self.checkpoint_dir = list(path.rglob("*.pt"))

    def train(self,model, max_iters, max_moves, num_nodes, num_agents, seed):
        trainer = decentralized_training.dpmrl_trainer(model = model, max_iters = max_iters,seed=seed, saving_dir = "./checkpoints",max_moves=max_moves, training_id = self.training_id)
        trainer.train_loop(num_nodes = num_nodes, num_agents = num_agents, random_seed = seed)

    def eval(self, num_nodes,num_agents, checkpoint:str,model,max_iters, seed,max_moves):
        eval = algorithm_evaluator(num_moves = max_moves, seed = seed, num_agents=(num_agents), num_nodes = num_nodes)
        eval.full_model(num_nodes=num_nodes,num_agents=num_agents,ckpt = checkpoint, model=model,log_name=f"decentralized_data_collection_{model.__name__}_{seed}_{num_agents}",max_iters=max_iters,seed = seed)
    def full_loop(self):
        # Combinatorially tractable
        for model_obj in tqdm.tqdm(self.model_list):
            print(f"setting model object to {model_obj}")
            for num_nodes, num_agents in self.num_nodes_to_agents.items():
                print(f"setting num nodes {num_nodes}, agents to {num_agents}")
                for seed in self.random_seeds:
                    print(f"Setting seed to {seed}")
                    print("Starting training...")
                    print(self.max_moves)
                    print(self.max_iters)
                    self.train(model = model_obj, max_moves = self.max_moves, max_iters = self.max_iters, num_nodes = num_nodes, num_agents = num_agents, seed = seed)
                    print("Starting evaluation...")
        """new_checkpoint_ls:list = []
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
            self.eval(model = getattr(globals().get(full_checkpoint[3]),str(full_checkpoint[3])), num_nodes = full_checkpoint[2], num_agents = self.num_agents, max_iters = 20, seed = full_checkpoint[1], max_moves = self.max_moves, checkpoint = full_checkpoint[0])"""

if __name__ == "__main__":
    automaticTrainerTester = auto_trainer_tester()
    automaticTrainerTester.full_loop()
