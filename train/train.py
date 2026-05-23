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
from stable_baselines3 import PPO
import yaml
with open("configs/ant.yaml", "r") as f:
    config = yaml.safe_load(f)
if config["use_custom_reward"]:
    env = CustomAntWrapper(gym.make("Ant-v5"))
else:
    env = gym.make("Ant-v5")
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=config["learning_rate"],
    n_steps=config["n_steps"],
    batch_size=config["batch_size"],
    verbose=1,
    tensorboard_log="logs_default/"
)
model.learn(total_timesteps=config["total_timesteps"])
model.save("results/ppo_ant_default")