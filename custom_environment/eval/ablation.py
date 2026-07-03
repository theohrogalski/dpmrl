from models_extra_attention_d import models_extra_attention
from models_no_dcbf_d import models_no_dcbf
from custom_environment.decentralized_agents.model_variants.models_no_collision import models_no_collision
from models_no_dcbf_no_state_estimation_d import models_no_dcbsf_no_state_est
from custom_environment.decentralized_agents.model_variants.models_extra_attention_d import models_extra_attention
from custom_environment.decentralized_agents.model_variants.models_no_dcbf_d import models_no_dcbf
from custom_environment.decentralized_agents.model_variants.models_no_dcbf_no_state_estimation_d import models_no_dcbf_no_state_est
from custom_environment.decentralized_agents.model_variants.models_full_model_d import models_full_model
from custom_environment.decentralized_agents.model_variants.models_no_state_estimation import models_no_state_est
from custom_environment.training.training import trainer 
model_list = [models_no_dcbf,models_no_dcbsf_no_state_est,models_full_model,models_no_state_est,models_no_collision]
model_list_2=[models_extra_attention]
if __name__=="__main__":
    for model in model_list_2:
        print(model)

        training_object = trainer(model=model,max_iters=100,max_moves=500,saving_dir="ablation")
        
        nodes_for_data=[50]
        random_seeds=[103,878,422]
        num_agents_for_testing=[4]
        for nn in nodes_for_data:
            for ag in num_agents_for_testing:
                for random_seed in random_seeds:
                    training_object.train_loop(num_nodes=nn,num_agents=ag, random_seed=random_seed)
    print("done")