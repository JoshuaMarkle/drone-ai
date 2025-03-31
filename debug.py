# debug.py

import matplotlib.pyplot as plt
import csv
import numpy as np
import config

episodes = []
rewards = []
steps = []

with open(config.LOG_PATH, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        episodes.append(int(row["episode"]))
        steps.append(int(row["steps"]))
        rewards.append(float(row["reward"]))

episodes_np = np.array(episodes)
rewards_np = np.array(rewards)
steps_np = np.array(steps)

# Fit lines of best fit (degree 1 polynomial = line)
reward_fit = np.polyfit(episodes_np, rewards_np, 1)
reward_trend = np.poly1d(reward_fit)

step_fit = np.polyfit(episodes_np, steps_np, 1)
step_trend = np.poly1d(step_fit)

# Print equations
print(f"Reward trend line: y = {reward_fit[0]:.4f}x + {reward_fit[1]:.2f}")
print(f"Steps trend line: y = {step_fit[0]:.4f}x + {step_fit[1]:.2f}")

# Plot Rewards
plt.figure()
plt.plot(episodes, rewards, label="Episode Reward")
plt.plot(episodes, reward_trend(episodes_np), 'r--', label=f"Best Fit: y={reward_fit[0]:.2f}x+{reward_fit[1]:.2f}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward over Episodes")
plt.grid(True)
plt.legend()
plt.show()

# Plot Steps
# plt.figure()
# plt.plot(episodes, steps, label="Steps Survived", color='orange')
# plt.plot(episodes, step_trend(episodes_np), 'r--', label=f"Best Fit: y={step_fit[0]:.2f}x+{step_fit[1]:.2f}")
# plt.xlabel("Episode")
# plt.ylabel("Steps")
# plt.title("Steps Survived over Episodes")
# plt.grid(True)
# plt.legend()
# plt.show()
