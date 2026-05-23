"""
This module reads logged data to extract the reward during training.
Converts it to visualize PPO Ant training curve
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

log_dir = 'logs/PPO_1/'
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

rewards = event_acc.Scalars("rollout/ep_rew_mean")
steps = [r.step for r in rewards]
values = [r.value for r in rewards]

plt.plot(steps,values)
plt.title('PPO Ant Training Curve')
plt.ylabel('Rewards')
plt.xlabel('timesteps')
plt.savefig("notebooks/training_curve.png")