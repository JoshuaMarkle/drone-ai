import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import config
import random

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(PPOActor, self).__init__()
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.std_layer = nn.Linear(input_dim, action_dim)
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, state):
        x = self.hidden_layers(state)
        mean = torch.sigmoid(self.mean_layer(x))    # Output in [0, 1]
        std = F.softplus(self.std_layer(x)) + 0.1   # Positive std
        return mean, std

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_layers):
        super(PPOCritic, self).__init__()
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state).squeeze(-1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_layers, learning_rate, clip_epsilon=config.PPO_CLIP_EPSILON):
        self.actor = PPOActor(state_dim, action_dim, hidden_layers).to(device)
        self.critic = PPOCritic(state_dim, config.CRITIC_HIDDEN_LAYERS).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LEARNING_RATE)

        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = config.ENTROPY_COEFF
        self.max_grad_norm = config.MAX_GRAD_NORM

    def get_action(self, state):
        state = state.to(device)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample().clamp(0, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def predict_value(self, state):
        return self.critic(state.to(device))

    def update(self, states, actions, old_log_probs, advantages):
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)

        for _ in range(config.PPO_EPOCHS):
            indices = list(range(len(states)))
            random.shuffle(indices)

            for i in range(0, len(indices), config.MINIBATCH_SIZE):
                batch_idx = indices[i:i + config.MINIBATCH_SIZE]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]

                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                surrogate_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

                actor_loss = surrogate_loss - self.entropy_coeff * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

    def update_critic(self, states, returns):
        states = states.to(device)
        returns = returns.to(device)

        for _ in range(config.PPO_EPOCHS):
            indices = list(range(len(states)))
            random.shuffle(indices)

            for i in range(0, len(indices), config.MINIBATCH_SIZE):
                batch_idx = indices[i:i + config.MINIBATCH_SIZE]
                batch_states = states[batch_idx]
                batch_returns = returns[batch_idx]

                values = self.critic(batch_states)
                value_loss = F.mse_loss(values, batch_returns)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

if __name__ == "__main__":
    state_dim = config.STATE_DIM
    action_dim = config.ACTION_DIM
    hidden_layers = config.ACTOR_HIDDEN_LAYERS
    learning_rate = config.ACTOR_LEARNING_RATE

    ppo_agent = PPOAgent(state_dim, action_dim, hidden_layers, learning_rate)

    print("PPO Actor Network:")
    print(ppo_agent.actor)
    print("\nPPO Critic Network:")
    print(ppo_agent.critic)
