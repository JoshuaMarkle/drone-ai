# train.py

import torch
import math
import os
from drone import Drone
from target import Target
from neural import PPOAgent
import config
import numpy as np
import csv

def normalize_state(drone, target):
    dx, dy = target.position - drone.position
    vx, vy = drone.velocity
    theta = drone.angle % (2 * math.pi)
    omega = drone.angular_velocity

    return torch.tensor([
        dx / config.MAX_DISTANCE,
        dy / config.MAX_DISTANCE,
        vx / config.MAX_SPEED,
        vy / config.MAX_SPEED,
        theta / (2 * math.pi),
        omega / config.MAX_OMEGA
    ], dtype=torch.float32)

def compute_reward(drone, target, prev_dist, crashed):
    if crashed:
        return config.REWARD_CRASH

    dist = np.linalg.norm(target.position - drone.position)
    reward = config.REWARD_ALIVE
    reward += config.REWARD_DISTANCE * (prev_dist - dist)

    if dist < config.TARGET_THRESHOLD:
        reward += config.REWARD_TARGET

    return reward

def is_out_of_bounds(pos):
    x, y = pos
    return x < -2 or x > 4 or y < -2 or y > 4

def train():
    agent = PPOAgent(
        config.STATE_DIM,
        config.ACTION_DIM,
        config.ACTOR_HIDDEN_LAYERS,
        config.ACTOR_LEARNING_RATE
    )

    # Initialize log file
    with open(config.LOG_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "steps", "reward"])

    # Trajectory buffer
    all_states = []
    all_actions = []
    all_log_probs = []
    all_returns = []

    for episode in range(config.TRAIN_EPISODES):
        drone = Drone(position=[config.DRONE_START_X, config.DRONE_START_Y], angle=config.DRONE_START_THETA)
        target = Target(position=[config.TARGET_START_X, config.TARGET_START_Y])

        ep_states = []
        ep_actions = []
        ep_log_probs = []
        ep_rewards = []
        ep_returns = []

        for step in range(config.EPISODE_LENGTH):
            state = normalize_state(drone, target)
            dist = np.linalg.norm(target.position - drone.position)

            with torch.no_grad():
                value = agent.predict_value(state)

            action, log_prob = agent.get_action(state)

            left_thrust, right_thrust = action.detach().numpy()
            drone.update(config.DT, left_thrust, right_thrust)

            crashed = is_out_of_bounds(drone.position)
            reward = compute_reward(drone, target, dist, crashed)

            ep_states.append(state)
            ep_actions.append(action)
            ep_log_probs.append(log_prob.detach())
            ep_rewards.append(reward)

            if crashed or dist < config.TARGET_THRESHOLD:
                break

        # Compute TD(1) returns: r + γ * V(s')
        with torch.no_grad():
            final_state = normalize_state(drone, target)
            final_value = agent.predict_value(final_state).item() if not crashed else 0.0

        G = final_value
        for r in reversed(ep_rewards):
            G = r + config.DISCOUNT_FACTOR * G
            ep_returns.insert(0, G)

        # Append to full buffer
        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_log_probs.extend(ep_log_probs)
        all_returns.extend(ep_returns)

        # Trigger update once buffer hits threshold
        if len(all_states) >= config.BUFFER_CAPACITY or episode == config.TRAIN_EPISODES - 1:
            # Convert to tensors
            states = torch.stack(all_states)
            actions = torch.stack(all_actions)
            log_probs = torch.stack(all_log_probs)
            returns = torch.tensor(all_returns, dtype=torch.float32)
            values = agent.critic(states).detach()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Train
            agent.update(states, actions, log_probs, advantages)
            agent.update_critic(states, returns)

            # Reset buffer
            all_states.clear()
            all_actions.clear()
            all_log_probs.clear()
            all_returns.clear()

        # Log this episode
        with open(config.LOG_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, step, sum(ep_rewards)])

        if episode % config.DEBUG_EPISODE_MARKER == 0:
            print(f"Episode {episode} | Steps: {step} | Reward: {sum(ep_rewards):.2f}")

    # Save model
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(config.MODEL_SAVE_PATH, "actor.pt"))
    print("Model saved.")

train()
