# config.py

# Starting state
DRONE_START_X = 0.0
DRONE_START_Y = 0.0
DRONE_START_THETA = 0.0
TARGET_START_X = 2.0
TARGET_START_Y = 2.0

# Drone (realistic I think)
DRONE_MASS = 1.5            	# Mass in kg
DRONE_MOI = 0.5             	# Moment of Inertia
DRONE_HEIGHT = 0.1          	# Approximate body thickness in meters.
WING_SPAN = 1               	# Distance between the two rotors in meters.
MAX_THRUST = 15.0           	# Maximum thrust per wing in Newtons.

	# Render
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ZOOM = 50.0                 	# 50 pixels per meter.
CAMERA_CENTER = (0.0, 0.0)  	# Fixed camera center in world coordinates.

# Physics
GRAVITY = 9.81                  # Gravity (m/s^2)
DT = 0.01                       # Change in time between steps

# Neural Network
STATE_DIM = 6                   # [x, y, vx, vy, theta, omega]
ACTION_DIM = 2                  # thrust commands for left and right wings
ACTOR_HIDDEN_LAYERS = [64, 64]  # Two hidden layers with 64 neurons each for the actor
CRITIC_HIDDEN_LAYERS = [64, 64] # Two hidden layers with 64 neurons each for the critic
ACTOR_LEARNING_RATE = 1e-4      # learning rate for actor network
CRITIC_LEARNING_RATE = 1e-4     # learning rate for critic network
MODEL_SAVE_PATH = "./models/"   # where to save the model

# Training Parameters
TRAIN_EPISODES = 100            # each episode is a full trajectory where the AI trys to control
EPISODE_LENGTH = 10000          # how long each training episode is (longer trajectories)
DISCOUNT_FACTOR = 0.90          # discount for future rewards, value of 1 means long term value of 0.9-0 is short term goals
EXPLORATION_NOISE = 0.2         # noise added during exploration of new actions

BOTTOM_LIMIT = -2.0             # Y-position below which the drone "crashes"
CRASH_PENALTY = 1000.0          # Large penalty for crashing
ALIVE_REWARD = -0.1             # Small reward for staying alive each timestep
DISTANCE_PENALTY_WEIGHT = 1.0   # Weight factor for distance-based punishment
TARGET_THRESHOLD = 0.2          # Distance threshold to consider reaching the target
TARGET_REWARD = -500.0          # Large reward for reaching the target

# Debugging
DEBUG_EPISODE_MARKER = 1        # Distance between printed episode markers
