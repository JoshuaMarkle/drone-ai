import torch
import torch.nn as nn
import numpy as np
import pygame
import math

import config
from neural import ActorCritic
from drone import Drone
from target import Target
from simulate import render_simulation

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("Drone Training Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# Initialize rendering options
VISUALIZE_TRAINING = True  # Set to False to disable rendering
zoom_value = config.ZOOM

def get_state_tensor(drone):
    """
    Convert drone state dictionary into a torch tensor of shape (1, STATE_DIM).
    The state vector consists of:
        [x position, y position, x velocity, y velocity, angle, angular velocity]
    """
    state = drone.state()
    state_vec = np.concatenate([state["position"],
                                state["velocity"],
                                [state["angle"], state["angular_velocity"]]])
    return torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

def simulate_trajectory(drone, actor, add_noise_first=False):
    """
    Simulate one trajectory using the given drone and actor.
    Returns the cumulative cost over the episode.
    
    For each time step:
      - Get current state.
      - Compute action from actor network.
      - Optionally add noise to encourage exploration.
      - Update drone physics.
      - Compute cost based on distance to target and penalties.
      - Render the simulation if VISUALIZE_TRAINING is True.
    """
    cumulative_cost = 0.0
    target_pos = np.array([config.TARGET_START_X, config.TARGET_START_Y])

    for t in range(config.EPISODE_LENGTH):
        state_tensor = get_state_tensor(drone)
        # Compute action from the actor network (deterministic policy)
        action = actor.actor(state_tensor).squeeze(0).detach().numpy()

        # Add exploration noise only at the first timestep if enabled
        if t == 0 and add_noise_first:
            noise = np.random.normal(0, config.EXPLORATION_NOISE, size=action.shape)
            action += noise

        # Update the drone with the computed action
        drone.update(config.DT, left_thrust=action[0], right_thrust=action[1])

        pos = drone.state()["position"]
        # Compute cost based on the distance from the target
        cost = config.DISTANCE_PENALTY_WEIGHT * np.linalg.norm(pos - target_pos)

        # Check for crash condition
        if pos[1] < config.BOTTOM_LIMIT:
            cost += config.CRASH_PENALTY
            break  

        # Check if the drone has reached the target
        if np.linalg.norm(pos - target_pos) < config.TARGET_THRESHOLD:
            cost += config.TARGET_REWARD  
            break  

        cumulative_cost += cost

        # Pygame rendering
        if VISUALIZE_TRAINING:
            # Create a temporary target object for rendering
            target_obj = Target(position=[config.TARGET_START_X, config.TARGET_START_Y])
            render_simulation(screen, clock, font, drone, target_obj, action[0], action[1])
    
    # Estimate future cost from the final state using the critic network
    final_state_tensor = get_state_tensor(drone)
    with torch.no_grad():
        value_final = actor.critic(final_state_tensor).item()
    cumulative_cost += (config.DISCOUNT_FACTOR ** config.EPISODE_LENGTH) * value_final

    return cumulative_cost

def main():
    # Initialize the ActorCritic model with parameters from config.py
    ac = ActorCritic(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        actor_hidden_layers=config.ACTOR_HIDDEN_LAYERS,
        critic_hidden_layers=config.CRITIC_HIDDEN_LAYERS,
        actor_lr=config.ACTOR_LEARNING_RATE,
        critic_lr=config.CRITIC_LEARNING_RATE
    )
    
    # Training loop over episodes
    for episode in range(config.TRAIN_EPISODES):
        # Reset initial state for both trajectories
        pos = np.array([0.0, 0.0])
        vel = np.array([0.0, 0.0])
        angle, omega = 0.0, 0.0
        
        # Create two drone instances for evaluation with and without initial noise
        drone_A = Drone(position=pos, angle=angle, mass=1.5, moment_of_inertia=0.05)
        drone_A.velocity = vel.copy()
        drone_A.angular_velocity = omega

        drone_B = Drone(position=pos.copy(), angle=angle, mass=1.5, moment_of_inertia=0.05)
        drone_B.velocity = vel.copy()
        drone_B.angular_velocity = omega

        # Simulate two trajectories: one with deterministic actions and one with initial noise for exploration
        cost_A = simulate_trajectory(drone_A, ac, add_noise_first=False)
        cost_B = simulate_trajectory(drone_B, ac, add_noise_first=True)

        # Choose the trajectory with lower cumulative cost as the target for learning
        best_cost = min(cost_A, cost_B)
        better = "exploration" if cost_B < cost_A else "deterministic"

        # Prepare the initial state tensor for the network updates
        init_state = np.concatenate([pos, vel, [angle, omega]])
        init_state_tensor = torch.tensor(init_state, dtype=torch.float32).unsqueeze(0)
        
        # Compute the critic's value estimate and the loss against the best cost observed
        value_estimate = ac.critic(init_state_tensor)
        target_value = torch.tensor([[best_cost]], dtype=torch.float32)
        critic_loss = nn.MSELoss()(value_estimate, target_value)

        # Update critic network parameters
        ac.critic_optimizer.zero_grad()
        critic_loss.backward()
        ac.critic_optimizer.step()

        # Compute actor loss (using critic's evaluation) and update actor network parameters
        actor_loss = -ac.critic(init_state_tensor)
        ac.actor_optimizer.zero_grad()
        actor_loss.backward()
        ac.actor_optimizer.step()

        # Print status every 100 episodes
        if episode % config.DEBUG_EPISODE_MARKER == 0:
            print(f"Episode {episode:03d}: cost_A = {cost_A:.2f}, cost_B = {cost_B:.2f}, chosen = {better}, "
                  f"critic_loss = {critic_loss.item():.2f}, actor_loss = {actor_loss.item():.2f}")

    # Save trained models to disk
    torch.save(ac.actor.state_dict(), "actor_model.pth")
    torch.save(ac.critic.state_dict(), "critic_model.pth")
    print("Training completed and models saved.")

if __name__ == "__main__":
    main()
