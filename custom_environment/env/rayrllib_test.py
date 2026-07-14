from ray import tune
from ray.rllib.env import PettingZooEnv
from custom_environment_p import SimpleGridWorld
from ray.rllib.algorithms.ppo import PPO

def env_creator(config):
    
    return SimpleGridWorld(
        grid_size=config.get("grid_size", (10,10)),
        num_agents=config.get("num_agents", 1),
        max_cycles=config.get("max_cycles", 100),
    )
SimpleGridWorld().observation_space()["agent_0"]

ppo_config = {
    "env": "simple_grid",       
    "env_config": {            
        "grid_size": (10, 10),
        "num_agents": 1,
        "max_cycles": 100,
    },
    
    "multiagent": {
        "policies": {
            "shared_policy": (
                None,
                
                SimpleGridWorld().observation_space["agent_0"],
                SimpleGridWorld().action_space["agent_0"],
                {}
            ),
        },
        "policy_mapping_fn": lambda agent_id, **kwargs: "shared_policy",
    },
    "framework": "torch", 
    "num_workers": 1,
    "lr": 1e-3,
}
if __name__ == "__main__":
    tune.run(
        PPO,
        name="pettingzoo_parallel_to_rllib",
        stop={"episode_reward_mean": 0}, 
        config=ppo_config,
        local_dir="~/ray_results/",
    )