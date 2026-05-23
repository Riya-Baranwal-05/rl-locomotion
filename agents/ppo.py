"""
This module customizes the default PPO algorithm
uisng the config fle. It changes hyperparametrs like
learning rate, n_steps and batch_size.
"""


from stable_baselines3 import PPO
def create_ppo(env,config):
    model = PPO(policy='MlpPolicy',
                env=env,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                batch_size=config["batch_size"],
                verbose=1,
                tensorboard_log="logs/")

    return model