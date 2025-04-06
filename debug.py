# debug.py

import matplotlib.pyplot as plt
import csv
from collections import defaultdict

STEP_LOG_PATH = "./logs/step_rewards.csv"

def plot_detailed_step_rewards():
    dist_data = defaultdict(list)
    angle_data = defaultdict(list)
    total_data = defaultdict(list)

    with open(STEP_LOG_PATH, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            episode = int(row["episode"])
            step = int(row["step"])
            dist_reward = float(row["dist_reward"])
            angle_reward = float(row["angle_reward"])
            total_reward = float(row["total_reward"])

            dist_data[episode].append((step, dist_reward))
            angle_data[episode].append((step, angle_reward))
            total_data[episode].append((step, total_reward))

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for episode in sorted(total_data.keys()):
        color = plt.cm.viridis(episode / max(total_data.keys()))
        # Distance Reward
        steps, rewards = zip(*sorted(dist_data[episode]))
        axs[0].plot(steps, rewards, label=f"Ep {episode}", color=color)
        # Angle Reward
        steps, rewards = zip(*sorted(angle_data[episode]))
        axs[1].plot(steps, rewards, color=color)
        # Total Reward
        steps, rewards = zip(*sorted(total_data[episode]))
        axs[2].plot(steps, rewards, color=color)

    axs[0].set_title("Distance Reward vs Step")
    axs[1].set_title("Angle Reward vs Step")
    axs[2].set_title("Total Reward vs Step")
    axs[2].set_xlabel("Step")

    for ax in axs:
        ax.grid(True)
        ax.set_ylabel("Reward")

    axs[0].legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

plot_detailed_step_rewards()
