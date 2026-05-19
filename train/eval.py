"""
This module evaluated the trained model. It loads the already trained 
model, takes the current state of ant using .reset() function. It starts 
the loop to evaluate. The loop gives observation to mode, gets action.
Feeds this action using .step() function in the enviroment and gets new 
state of the ant through observation and reward . In the end total_reward 
stores reward throughout the loop.
Using terminated means the ant fell or truncated means time limit
set by MuJoCo's default Ant environment reached , both stops the 
loops
"""


import sys
sys.path.append(".")
import gymnasium as gym
from envs.custom_wrapper import CustomAntWrapper
from stable_baselines3 import PPO
env = CustomAntWrapper(gym.make("Ant-v5",render_mode="human"))
model = PPO.load("results/ppo_ant")
observation , info = env.reset()
total_reward = 0
terminated = False
truncated = False
while not (terminated or truncated):
    action,_states=model.predict(observation,deterministic=True)
    observation,reward,terminated,truncated,info = env.step(action)
    total_reward += reward
    env.render()

print("total_reward: ",total_reward )