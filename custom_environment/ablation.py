from custom_environment.model_variants.models_extra_attention_d import models_extra_attention
from custom_environment.model_variants.models_no_dcbf_d import models_no_dcbf
from custom_environment.model_variants.models_no_dcbf_no_state_estimation_d import models_no_dcbf_no_state_est
from models_full_model_d import models_full_model
from custom_environment.model_variants.models_no_state_estimation import models_no_state_est
from training import trainer 
model_list = [models_no_dcbf,models_no_dcbf_no_state_est,models_full_model,models_no_state_est]

if __name__=="__main__":
    for model in model_list:
        print(model)
        training_object = trainer(model= model)
        nodes_for_data=[50]
        num_agents_for_testing=[4]
        for nn in nodes_for_data:
            for ag in num_agents_for_testing:
                training_object.train_loop(num_nodes=nn,num_agents=ag)
    print("done")