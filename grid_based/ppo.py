from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from grid_based.custom_environment_p import GridWithMemory
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec

def env_creator(config):
    base_env = GridWithMemory()
    return PettingZooEnv(base_env)


register_env("grid_env", env_creator)

config = (
    PPOConfig()
    .environment("grid_env")
    .framework("torch") 
    .multi_agent(
        policies={"default_policy": PolicySpec()},
        policy_mapping_fn=lambda agent_id, *args: "default_policy"
    )
)

tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(stop={"training_iteration": 100})
).fit()