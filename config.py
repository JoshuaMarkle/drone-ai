# config.py

# Starting state
DRONE_START_X = 0.0
DRONE_START_Y = 0.0
DRONE_START_THETA = 0.0
TARGET_START_X = 2.0
TARGET_START_Y = 2.0

# Drone (realistic I think)
DRONE_MASS = 1.5            # Mass in kg
DRONE_MOI = 0.05            # Moment of Inertia
DRONE_HEIGHT = 0.1          # Approximate body thickness in meters.
WING_SPAN = 1               # Distance between the two rotors in meters.
MAX_THRUST = 15.0           # Maximum thrust per wing in Newtons.

# Render
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ZOOM = 50.0                 # 50 pixels per meter.
CAMERA_CENTER = (0.0, 0.0)  # Fixed camera center in world coordinates.

# Physics
GRAVITY = 9.81              # Gravity (m/s^2)
