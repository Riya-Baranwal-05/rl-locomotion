## DEMO
<!--GIF-->

# RL-LOCOMOTION AGENT TRAINING IN MUJOCO
This project will train MuJoCo's simulated ant to walk learning through PPO.

# What does this project does?
It trains the ant to walk. It uses forward velocity and stability as reward. 
For penalty it uses falling and energy efficiency. Actions are the joint angles/torques.
The policy is trained using neural network
giving states as inputs and outputs are the actions.

# Project Structure
- envs/
    --- custom_wrapper.py #reward shaping, obs normalization
- agents/
    --- ppo.py # PPO Implementation
    --- sac.py #SAC alternative
- train/
    --- train.py # main training loop
    --- eval.py # evaluation and rendering
- configs/
    --- ant.yaml #hyperparameters
    --- cheetah.yaml 
- results/ #saved models,training curve
- requirements.txt
- README.md

## How to Run
TODO

## What i learned
TODO

## Limitations
- Training is happening in simulation so the training might not work on real hardware. 
- In Simulations even , training steps are expensive on computation.
- RL is noisy , with same settings results can be different.

## References

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347
- MuJoCo (Todorov et al., 2012)
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, A., & Dormann, N. (2021). ,Stable-Baselines3: Reliable Reinforcement Learning Implementations . Journal of Machine Learning Research . (https://jmlr.org/papers/v22/20-1364.html)

