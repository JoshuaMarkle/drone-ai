# renderer.py
import pygame
import sys
import math
import numpy as np
from physics import Drone, Target
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, ZOOM, CAMERA_CENTER,
    DRONE_START_X, DRONE_START_Y, DRONE_START_THETA, TARGET_START_X, TARGET_START_Y,
    MAX_THRUST, WING_SPAN, DRONE_HEIGHT
)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Drone Simulation")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
zoom_value = ZOOM

# Create physics objects
drone = Drone(position=[DRONE_START_X, DRONE_START_Y], angle=DRONE_START_THETA)
target = Target(position=[TARGET_START_X, TARGET_START_Y])

def world_to_screen(pos, zoom, screen_width, screen_height, center):
    """
    Convert world coordinates to screen coordinates.
    
    Args:
        pos (iterable): (x, y) in world space.
        zoom (float): Scale factor.
        screen_width (int): Screen width in pixels.
        screen_height (int): Screen height in pixels.
        center (tuple): World coordinate to center the screen on.
        
    Returns:
        tuple: (x, y) in screen coordinates.
    """
    x, y = pos[0], pos[1]
    cx, cy = center[0], center[1]
    screen_x = (x - cx) * zoom + screen_width / 2
    screen_y = screen_height / 2 - (y - cy) * zoom
    return (int(screen_x), int(screen_y))

def draw_drone(surface, drone, zoom, center):
    """
    Draw the drone as a blue rectangle.
    
    The rectangle is defined in the drone's local coordinates and then transformed
    by the drone's position and orientation.
    """
    pos = drone.position
    angle = drone.angle
    w = WING_SPAN
    h = DRONE_HEIGHT
    local_points = np.array([
        [-w/2, -h/2],
        [ w/2, -h/2],
        [ w/2,  h/2],
        [-w/2,  h/2]
    ])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated_points = []
    for p in local_points:
        rx = p[0] * cos_a - p[1] * sin_a
        ry = p[0] * sin_a + p[1] * cos_a
        rotated_points.append((rx + pos[0], ry + pos[1]))
    screen_points = [world_to_screen(pt, zoom, SCREEN_WIDTH, SCREEN_HEIGHT, center) for pt in rotated_points]
    pygame.draw.polygon(surface, (0, 0, 255), screen_points)

def draw_thrust(surface, drone, left_thrust, right_thrust, zoom, center):
    """
    Visualize the thrust applied at each wing as red lines.
    """
    # Define local positions of the wings.
    left_wing_local = np.array([-WING_SPAN/2, 0])
    right_wing_local = np.array([WING_SPAN/2, 0])
    
    cos_a = math.cos(drone.angle)
    sin_a = math.sin(drone.angle)
    def local_to_world(local):
        x = local[0] * cos_a - local[1] * sin_a
        y = local[0] * sin_a + local[1] * cos_a
        return np.array([x, y]) + drone.position
    left_wing_world = local_to_world(left_wing_local)
    right_wing_world = local_to_world(right_wing_local)
    
    # Thrust direction: drone's local downward direction.
    thrust_dir = np.array([math.sin(drone.angle), -math.cos(drone.angle)])
    vis_scale = 0.1  # Visualization scaling factor.
    left_end = left_wing_world + thrust_dir * left_thrust * vis_scale
    right_end = right_wing_world + thrust_dir * right_thrust * vis_scale
    
    pygame.draw.line(surface, (255, 0, 0),
                     world_to_screen(left_wing_world, zoom, SCREEN_WIDTH, SCREEN_HEIGHT, center),
                     world_to_screen(left_end, zoom, SCREEN_WIDTH, SCREEN_HEIGHT, center), 2)
    pygame.draw.line(surface, (255, 0, 0),
                     world_to_screen(right_wing_world, zoom, SCREEN_WIDTH, SCREEN_HEIGHT, center),
                     world_to_screen(right_end, zoom, SCREEN_WIDTH, SCREEN_HEIGHT, center), 2)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Use mouse wheel to adjust zoom.
        if event.type == pygame.MOUSEWHEEL:
            zoom_value += event.y * 5.0
            if zoom_value < 5:
                zoom_value = 5

    # Handle key inputs for drone control.
    keys = pygame.key.get_pressed()
    left_thrust = 0.0
    right_thrust = 0.0
    if keys[pygame.K_UP]:
        left_thrust = MAX_THRUST
        right_thrust = MAX_THRUST
    elif keys[pygame.K_LEFT]:
        left_thrust = MAX_THRUST
    elif keys[pygame.K_RIGHT]:
        right_thrust = MAX_THRUST

    # Update the drone physics.
    drone.update(0.01, left_thrust, right_thrust)

    # Clear screen.
    screen.fill((255, 255, 255))
    
    # Draw target as a red circle.
    target_screen = world_to_screen(target.position, zoom_value, SCREEN_WIDTH, SCREEN_HEIGHT, CAMERA_CENTER)
    pygame.draw.circle(screen, (255, 0, 0), target_screen, 5)
    
    # Draw the drone and its thrust visualization.
    draw_drone(screen, drone, zoom_value, CAMERA_CENTER)
    draw_thrust(screen, drone, left_thrust, right_thrust, zoom_value, CAMERA_CENTER)
    
    # Draw debugging info (position, velocity, angle, angular velocity) at the top-left.
    state = drone.state()
    debug_text = (f"x: {state['position'][0]:.2f}  y: {state['position'][1]:.2f}  "
                  f"vx: {state['velocity'][0]:.2f}  vy: {state['velocity'][1]:.2f}  "
                  f"theta: {state['angle']:.2f}  omega: {state['angular_velocity']:.2f}")
    debug_surface = font.render(debug_text, True, (0, 0, 0))
    screen.blit(debug_surface, (10, 10))
    
    pygame.display.flip()
    clock.tick(60)
