"""
This creates Mujoco Ant environment in gymnasium and 
it conditionally uses the custom wrapper. It applies specific policy PPO 
for training. MlpPolicy is actually Multi Layer Perceptron neural
network. Later i train the model using .learn() and save it in results.

"""
import sys
sys.path.append(".")
from envs.custom_wrapper import CustomAntWrapper
import gymnasium as gym
from agents.ppo import create_ppo
import yaml
with open("configs/ant.yaml", "r") as f:
    config = yaml.safe_load(f)
if config["use_custom_reward"]:
    env = CustomAntWrapper(gym.make("Ant-v5"))
else:
    env = gym.make("Ant-v5")
model = create_ppo(env, config)
model.learn(total_timesteps=config["total_timesteps"])
model.save("results/ppo_ant_default")