import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        """
        Build a PPO Actor network that maps states to actions.

        Args:
            state_dim (int): Dimensionality of the input state.
            action_dim (int): Dimensionality of the output action.
            hidden_layers (list of int): List with the number of nodes for each hidden layer.
        """
        super(PPOActor, self).__init__()
        layers = []
        input_dim = state_dim  # Correct input size

        # Build hidden layers.
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim  # Update input_dim for the next layer

        # Output layers for action mean and standard deviation
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.std_layer = nn.Linear(input_dim, action_dim)

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, state):
        """ Forward pass through the network. """
        x = self.hidden_layers(state)  # Ensure input shape is correct
        mean = torch.sigmoid(self.mean_layer(x))  # Ensure output is between 0 and 1
        std = F.softplus(self.std_layer(x)) + 0.1  # Ensure standard deviation is positive
        return mean, std

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_layers, learning_rate, clip_epsilon=0.2):
        """
        Initialize the PPO agent with an Actor network.

        Args:
            state_dim (int): Dimension of the state vector.
            action_dim (int): Dimension of the action vector.
            hidden_layers (list of int): Hidden layer sizes for the actor network.
            learning_rate (float): Learning rate for the optimizer.
            clip_epsilon (float): Clipping parameter for PPO.
        """
        self.actor = PPOActor(state_dim, action_dim, hidden_layers)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.clip_epsilon = clip_epsilon

    def get_action(self, state):
        """ Sample an action based on the policy. """
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample().clamp(0, 1)  # Clamp sampled action to [0, 1]
        log_prob = dist.log_prob(action)
        return action, log_prob

    def update(self, states, actions, old_log_probs, advantages):
        """ Perform PPO policy update. """
        mean, std = self.actor(states)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        loss = -torch.min(ratio * advantages.unsqueeze(-1), clipped_ratio * advantages.unsqueeze(-1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    # Quick test: instantiate the PPO agent using the parameters from config.py.
    state_dim = config.STATE_DIM
    action_dim = config.ACTION_DIM
    hidden_layers = config.ACTOR_HIDDEN_LAYERS
    learning_rate = config.ACTOR_LEARNING_RATE

    # Create a PPOAgent instance.
    ppo_agent = PPOAgent(state_dim, action_dim, hidden_layers, learning_rate)

    # Print network summary.
    print("PPO Actor Network:")
    print(ppo_agent.actor)
