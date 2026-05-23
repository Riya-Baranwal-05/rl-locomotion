"""
This creates Mujoco Ant environment in gymnasium and modifies
it using the custom wrapper. It applies specific policy PPO 
for training. MlpPolicy is actually Multi Layer Perceptron neural
 network. Later i train the model using .learn() and save it in results.

"""
import sys
sys.path.append(".")
from envs.custom_wrapper import CustomAntWrapper
import gymnasium as gym
from stable_baselines3 import PPO

env = CustomAntWrapper(gym.make("Ant-v5"))
model = PPO(policy="MlpPolicy",env=env,verbose=1,tensorboard_log="logs/")
model.learn(total_timesteps=1000000)
model.save("results/ppo_ant_v2")