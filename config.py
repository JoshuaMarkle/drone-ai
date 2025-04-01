# config.py

# Starting state
DRONE_START_X = 0.0
DRONE_START_Y = 0.0
DRONE_START_THETA = 0.0
TARGET_START_X = 2.0
TARGET_START_Y = 2.0

# Drone
DRONE_MASS = 1.5            	# Mass in kg
DRONE_MOI = 0.5             	# Moment of Inertia
DRONE_HEIGHT = 0.1          	# Approximate body thickness in meters.
WING_SPAN = 1               	# Distance between the two rotors in meters.
MAX_THRUST = 20.0           	# Maximum thrust per wing in Newtons.

MAX_SPEED = 30.0                # Used for normilization (m/s)
MAX_DISTANCE = 5.0              # Used for normilization (m)
MAX_OMEGA = 50.0                # Used for normilization (rad/s)

# Render
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ZOOM = 50.0                 	# 50 pixels per meter.
CAMERA_CENTER = (0.0, 0.0)  	# Fixed camera center in world coordinates.

# Physics
GRAVITY = 9.81                  # Gravity (m/s^2)
DT = 0.01                       # Change in time between steps

# Neural Network
STATE_DIM = 6                   # [dist_x, dist_y, vx, vy, theta, omega]
ACTION_DIM = 2                  # left & right thrust (between 0-1)
ACTOR_HIDDEN_LAYERS = [64, 64]    # Hidden layers for the actor
CRITIC_HIDDEN_LAYERS = [64, 64]   # Hidden layers for the critic
ACTOR_LEARNING_RATE = 1e-4      # learning rate for actor network
CRITIC_LEARNING_RATE = 1e-4     # learning rate for critic network
MODEL_SAVE_PATH = "./models/"   # where to save the model

PPO_EPOCHS = 10                 # Number of times to reuse buffer per update
MINIBATCH_SIZE = 64             # Minibatch size for actor & critic updates
PPO_CLIP_EPSILON = 0.1          # Clipping parameter for ratio
ENTROPY_COEFF = 0.01            # Weight for entropy regularization
MAX_GRAD_NORM = 0.5             # Gradient clipping max norm
BUFFER_CAPACITY = 4096          # Number of transitions to collect before update

# Training Parameters
TRAIN_EPISODES = 100000         # each episode is a full trajectory
EPISODE_LENGTH = 500            # how long each training episode is (longer trajectories)
DISCOUNT_FACTOR = 0.99          # discount for future rewards, between 0.99 and 0.9
EXPLORATION_NOISE = 0.05        # noise added during exploration of new actions

REWARD_CRASH = -100.0           # Large penalty for crashing
REWARD_ALIVE = 1.0              # Small reward for staying alive each timestep
REWARD_DISTANCE = 3.0           # Reward/punishment for change of distance between timesteps
REWARD_TARGET = 100.0           # Large reward for reaching the target
REWARD_ANGLE = 2.0              # Reward staying upright
TARGET_THRESHOLD = 0.2          # Distance threshold to consider reaching the target

# Debugging
DEBUG_EPISODE_MARKER = 50       # Distance between printed episode markers
CHECKPOINT_MARKER = 500         # Distance between saved model checkpoints
LOG_PATH = "./logs/log.csv"     # Path to the log file
CHECKPOINT_PATH = "./models/checkpoints/"   # where to save the model
