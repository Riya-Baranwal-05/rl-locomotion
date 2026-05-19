"""
This creates Mujoco Ant environment in gymnasium and modifies
it using the custom wrapper. It applies specific policy PPO 
for training. MlpPolicy is actually Multi Layer Perceptron neural
 network. Later i train the model using .learn() and save it in results.

"""
from envs import custom_wrapper
import gymnasium as gym
from stable_baselines3 import PPO

env = custom_wrapper.CustomAntWrapper(gym.make("Ant-v5"))
model = PPO(policy="MlpPolicy",env=env)
model.learn(total_timesteps=100000)
model.save("results/ppo_ant")