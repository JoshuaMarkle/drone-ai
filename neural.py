import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config

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
        mean = torch.sigmoid(self.mean_layer(x))      # Output between 0 and 1
        std = F.softplus(self.std_layer(x)) + 0.1      # Ensure positive std
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

        layers.append(nn.Linear(input_dim, 1))  # Output = state value
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state).squeeze(-1)  # Return shape (batch,)

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_layers, learning_rate, clip_epsilon=0.2):
        self.actor = PPOActor(state_dim, action_dim, hidden_layers)
        self.critic = PPOCritic(state_dim, config.CRITIC_HIDDEN_LAYERS)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LEARNING_RATE)

        self.clip_epsilon = clip_epsilon

    def get_action(self, state):
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().clamp(0, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def predict_value(self, state):
        return self.critic(state)

    def update(self, states, actions, old_log_probs, advantages):
        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # PPO surrogate loss
        surrogate_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        self.actor_optimizer.zero_grad()
        surrogate_loss.backward()
        self.actor_optimizer.step()

    def update_critic(self, states, returns):
        values = self.critic(states)
        loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        loss.backward()
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
