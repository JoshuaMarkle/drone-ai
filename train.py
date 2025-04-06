# train.py

import torch
import math
import os
import shutil
from drone import Drone
from target import Target
from neural import PPOAgent
import config
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)

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
    ], dtype=torch.float32, device=device)

def compute_reward(drone, target, prev_dist, crashed):
    # Reward for staying alive
    reward = 0
    reward += config.REWARD_ALIVE

    # Reward for getting closer to the target
    dist = np.linalg.norm(target.position - drone.position)
    reward += config.REWARD_DISTANCE * (prev_dist - dist)

    # Normalize angle to [-pi, pi] (reward max: upright and -max: upsidedown)
    angle = ((drone.angle + math.pi) % (2 * math.pi)) - math.pi
    reward += config.REWARD_ANGLE * math.cos(angle)

    # If the drone is on the target, add another additional reward
    if dist < config.TARGET_THRESHOLD:
        reward += config.REWARD_TARGET

    # If the drone crashed, give a penalty
    if crashed:
        reward += config.REWARD_CRASH

    return reward

def is_out_of_bounds(pos):
    x, y = pos
    return x < -2 or x > 4 or y < -2 or y > 4

def train():
    if os.path.exists(config.CHECKPOINT_PATH):
        shutil.rmtree(config.CHECKPOINT_PATH)
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(config.MODEL_PATH, exist_ok=True)

    writer = SummaryWriter()

    agent = PPOAgent(
        config.STATE_DIM,
        config.ACTION_DIM,
        config.ACTOR_HIDDEN_LAYERS,
        config.ACTOR_LEARNING_RATE
    )

    # CSV log setup
    # with open(config.LOG_PATH, mode='w', newline='') as file:
    #     writer_csv = csv.writer(file)
    #     writer_csv.writerow(["episode", "steps", "reward", "critic_start"])

    # Log reward over the course of the episode
    with open(config.STEP_LOG_PATH, mode='w', newline='') as f:
        writer_step = csv.writer(f)
        writer_step.writerow(["episode", "step", "dist_reward", "angle_reward", "total_reward"])

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
                _ = agent.predict_value(state)

            action, log_prob = agent.get_action(state)
            left_thrust, right_thrust = action.detach().cpu().numpy()
            drone.update(config.DT, left_thrust, right_thrust)

            crashed = is_out_of_bounds(drone.position)
            reward = compute_reward(drone, target, dist, crashed)

            ep_states.append(state)
            ep_actions.append(action)
            ep_log_probs.append(log_prob.detach())
            ep_rewards.append(reward)

            # Log reward over episode to CSV file (every checkpoint marker amount of episodes)
            if config.EPISODE_MARKER > 0 and episode % config.EPISODE_MARKER == 0 and episode > 0:
                with open(config.STEP_LOG_PATH, mode='a', newline='') as f:
                    writer_step = csv.writer(f)

                    angle = ((drone.angle + math.pi) % (2 * math.pi)) - math.pi
                    prev_dist = np.linalg.norm(target.position - drone.position)
                    angle_reward = config.REWARD_ANGLE * math.cos(angle)
                    dist_reward = config.REWARD_DISTANCE * (prev_dist - dist)

                    writer_step.writerow([episode, step, dist_reward, angle_reward, reward])

            if crashed:
                break

        with torch.no_grad():
            final_state = normalize_state(drone, target)
            final_value = agent.predict_value(final_state).item() if not crashed else 0.0

        G = final_value
        for r in reversed(ep_rewards):
            G = r + config.DISCOUNT_FACTOR * G
            ep_returns.insert(0, G)

        with torch.no_grad():
            start_value = agent.predict_value(ep_states[0]).item()

        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_log_probs.extend(ep_log_probs)
        all_returns.extend(ep_returns)

        if len(all_states) >= config.BUFFER_CAPACITY or episode == config.TRAIN_EPISODES - 1:
            states = torch.stack(all_states)
            actions = torch.stack(all_actions)
            log_probs = torch.stack(all_log_probs)
            returns = torch.tensor(all_returns, dtype=torch.float32, device=device)
            values = agent.critic(states).detach()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            agent.update(states, actions, log_probs, advantages)
            agent.update_critic(states, returns)

            all_states.clear()
            all_actions.clear()
            all_log_probs.clear()
            all_returns.clear()

        ep_reward = sum(ep_rewards)

        # TensorBoard: only log 3 selected metrics
        writer.add_scalar("Reward/Total", ep_reward, episode)
        writer.add_scalar("Episode/Steps", step, episode)
        writer.add_scalar("Critic/StartValue", start_value, episode)

        # CSV: log only selected values
        # with open(config.LOG_PATH, mode='a', newline='') as file:
        #     writer_csv = csv.writer(file)
        #     writer_csv.writerow([episode, step, ep_reward, start_value])

        if episode % config.DEBUG_EPISODE_MARKER == 0:
            print(f"Ep {episode} | Steps: {step} | Reward: {ep_reward:.2f}")

        if config.CHECKPOINT_MARKER > 0 and episode % config.CHECKPOINT_MARKER == 0 and episode > 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f"actor_ep{episode}.pt")
            torch.save(agent.actor.state_dict(), checkpoint_path)

    torch.save(agent.actor.state_dict(), os.path.join(config.MODEL_SAVE_PATH, "actor.pt"))
    print("Model saved.")
    writer.close()

train()
