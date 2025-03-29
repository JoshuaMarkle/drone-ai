# neural.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        """
        Build an Actor network that maps states to actions.
        
        Args:
            state_dim (int): Dimensionality of the input state.
            action_dim (int): Dimensionality of the output action.
            hidden_layers (list of int): List with the number of nodes for each hidden layer.
        """
        super(ActorNetwork, self).__init__()
        layers = []
        input_dim = state_dim

        # Build hidden layers.
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Final layer maps to action_dim. Using Tanh to bound the actions (e.g., between -1 and 1).
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.model(state)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layers):
        """
        Build a Critic network that maps states to a scalar value.
        
        Args:
            state_dim (int): Dimensionality of the input state.
            hidden_layers (list of int): List with the number of nodes for each hidden layer.
        """
        super(CriticNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.model(state)

class ActorCritic:
    def __init__(self, state_dim, action_dim, actor_hidden_layers, critic_hidden_layers, actor_lr, critic_lr):
        """
        Initialize the Actor-Critic container.
        
        Args:
            state_dim (int): Dimension of the state vector.
            action_dim (int): Dimension of the action vector.
            actor_hidden_layers (list of int): Hidden layer sizes for the actor network.
            critic_hidden_layers (list of int): Hidden layer sizes for the critic network.
            actor_lr (float): Learning rate for the actor.
            critic_lr (float): Learning rate for the critic.
        """
        self.actor = ActorNetwork(state_dim, action_dim, actor_hidden_layers)
        self.critic = CriticNetwork(state_dim, critic_hidden_layers)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def act(self, state):
        """
        Given a state, return the action according to the actor network.
        Here the policy is deterministic.
        
        Args:
            state (torch.Tensor): A tensor representing the state.
        
        Returns:
            torch.Tensor: Action output from the actor.
        """
        return self.actor(state)
    
    def evaluate(self, state):
        """
        Given a state, return the value estimate from the critic network.
        
        Args:
            state (torch.Tensor): A tensor representing the state.
            
        Returns:
            torch.Tensor: Value estimate.
        """
        return self.critic(state)

if __name__ == "__main__":
    # Quick test: instantiate the networks using the parameters from config.py.
    state_dim = config.STATE_DIM
    action_dim = config.ACTION_DIM
    actor_hidden_layers = config.ACTOR_HIDDEN_LAYERS
    critic_hidden_layers = config.CRITIC_HIDDEN_LAYERS
    actor_lr = config.ACTOR_LEARNING_RATE
    critic_lr = config.CRITIC_LEARNING_RATE

    # Create an ActorCritic instance.
    ac = ActorCritic(state_dim, action_dim, actor_hidden_layers, critic_hidden_layers, actor_lr, critic_lr)
    
    # Print network summaries.
    print("Actor Network:\n", ac.actor)
    print("\nCritic Network:\n", ac.critic)
