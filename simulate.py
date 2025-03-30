import pygame
import sys
from drone import Drone
from target import Target
import config

# Setup PyGame
pygame.init()
screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
pygame.display.set_caption("Drone Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# Create the drone and target objects
drone = Drone(position=[config.DRONE_START_X, config.DRONE_START_Y], angle=config.DRONE_START_THETA)
target = Target(position=[config.TARGET_START_X, config.TARGET_START_Y])

# Toggle control mode: True = user control; False = (placeholder for NN control)
# True = user controls drone
# False = AI controls drone
user_control = True
def render_simulation(screen, clock, font, drone, target, left_thrust, right_thrust, training_info=None):
    # Draw everything
    screen.fill((255, 255, 255))
    target.draw(screen)
    drone.draw(screen, left_thrust, right_thrust)

    # Draw debugging info
    state = drone.state()
    lines = [
        f"x: {state['position'][0]:.2f}  y: {state['position'][1]:.2f}",
        f"vx: {state['velocity'][0]:.2f}  vy: {state['velocity'][1]:.2f}",
        f"θ: {state['angle']:.2f}  ω: {state['angular_velocity']:.2f}",
        f"thrust l: {left_thrust * config.MAX_THRUST:.2f}  r: {right_thrust * config.MAX_THRUST:.2f}"
    ]

    if training_info:
        training_lines = [
            f"episode: {training_info['episode']}",
            f"reward: {training_info['reward']:.2f}",
            f"distance: {training_info['distance']:.2f}",
            f"mode: {training_info['mode']}"
        ]
        lines.extend(training_lines)

    y_offset = 10
    for line in lines:
        debug_surface = font.render(line, True, (0, 0, 0))
        screen.blit(debug_surface, (10, y_offset))
        y_offset += debug_surface.get_height()

    # Wait for next frame
    pygame.display.flip()
    clock.tick(60)

if __name__ == "__main__":
    # Game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN: # Toggle control mode with [SPACE]
                if event.key == pygame.K_SPACE:
                    user_control = not user_control
            if event.type == pygame.MOUSEWHEEL: # Adjust zoom with [MOUSEWHEEL]
                new_zoom = config.ZOOM + event.y * 5.0
        
        left_thrust = 0.0
        right_thrust = 0.0
        if user_control: # USER CONTROLS
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
        else: # NEURAL NETWORK CONTROLS
            # PLACEHOLDER
            left_thrust = 0.0
            right_thrust = 0.0

        # Update drone physics
        drone.update(config.DT, left_thrust, right_thrust)

        render_simulation(screen, clock, font, drone, target, left_thrust, right_thrust)
