import gymnasium
import numpy as np
class CustomAntWrapper(gymnasium.Wrapper):
    def __init__(self,env):
        super().__init__(env)

    def step(self,action):
        """
        This modifies default reward from MuJoCo by adding additional 
        rewards like forward velocity of the ant and substracting energy penalty 
        falling penalty which will be tuned if needed
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        forward_velocity = info["x_velocity"]
        energy_penalty = np.sum(np.abs(action))
        fallen_penalty = 10 if  terminated else 0
        reward = reward + forward_velocity - energy_penalty - fallen_penalty

        return observation, reward,terminated,truncated,info

