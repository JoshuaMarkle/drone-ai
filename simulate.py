import pygame
import sys
import torch
from drone import Drone
from target import Target
from neural import PPOActor
import config
import math

# --- Initialize PyGame ---
pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("Drone Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# --- Create Drone + Target ---
drone = Drone(position=[config.DRONE_START_X, config.DRONE_START_Y], angle=config.DRONE_START_THETA)
target = Target(position=[config.TARGET_START_X, config.TARGET_START_Y])

# --- Load Trained Model ---
actor = PPOActor(config.STATE_DIM, config.ACTION_DIM, config.ACTOR_HIDDEN_LAYERS)
actor.load_state_dict(torch.load(config.CHECKPOINT_PATH + "actor_ep3000.pt"))
actor.eval()

# --- Normalize State ---
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

# --- Control Mode ---
user_control = False  # Default to AI control

# --- Render ---
def render_simulation(screen, clock, font, drone, target, left_thrust, right_thrust):
    screen.fill((255, 255, 255))
    target.draw(screen)
    drone.draw(screen, left_thrust, right_thrust)

    # Debug overlay
    state = drone.state()
    lines = [
        f"x: {state['position'][0]:.2f}  y: {state['position'][1]:.2f}",
        f"vx: {state['velocity'][0]:.2f}  vy: {state['velocity'][1]:.2f}",
        f"θ: {state['angle']:.2f}  ω: {state['angular_velocity']:.2f}",
        f"thrust l: {left_thrust * config.MAX_THRUST:.2f}  r: {right_thrust * config.MAX_THRUST:.2f}",
        f"mode: {'USER' if user_control else 'AI'}"
    ]

    y_offset = 10
    for line in lines:
        debug_surface = font.render(line, True, (0, 0, 0))
        screen.blit(debug_surface, (10, y_offset))
        y_offset += debug_surface.get_height()

    pygame.display.flip()
    clock.tick(60)

# --- Main Loop ---
if __name__ == "__main__":
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    user_control = not user_control
            if event.type == pygame.MOUSEWHEEL:
                new_zoom = config.ZOOM + event.y * 5.0

        left_thrust = 0.0
        right_thrust = 0.0

        if user_control:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                left_thrust = 1.0
                right_thrust = 1.0
            if keys[pygame.K_LEFT]:
                right_thrust = 1.0
                left_thrust /= 2.0
            if keys[pygame.K_RIGHT]:
                left_thrust = 1.0
                right_thrust /= 2.0
        else:
            with torch.no_grad():
                state = normalize_state(drone, target)
                mean, _ = actor(state)
                left_thrust = mean[0].item()
                right_thrust = mean[1].item()

        drone.update(config.DT, left_thrust, right_thrust)
        render_simulation(screen, clock, font, drone, target, left_thrust, right_thrust)
