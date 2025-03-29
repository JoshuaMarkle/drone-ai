import numpy as np
import math
import pygame
import config

class Drone:
    def __init__(self, position, angle=0.0, mass=config.DRONE_MASS, moment_of_inertia=config.DRONE_MOI):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.angle = angle
        self.angular_velocity = 0.0
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia

    def apply_thrust(self, thrust):
        """
        Compute the thrust force downwards from drone's point of view
        """
        fx = -thrust * math.sin(self.angle)
        fy = thrust * math.cos(self.angle)
        return np.array([fx, fy])

    def update(self, dt, left_thrust=0.0, right_thrust=0.0):
        """
        Update the drone's state (physics update) over timestep dt using left and right wing thrust commands.
        Expect left and right thrust to be from 0 to 1
        """
        # Calculate thrust values and total upwards force
        left_thrust *= config.MAX_THRUST
        right_thrust *= config.MAX_THRUST
        total_thrust = left_thrust + right_thrust

        # Calculate torque
        torque = -(left_thrust - right_thrust) * (config.WING_SPAN / 2)
        
        # Linear: net forces + velocity
        gravity_force = np.array([0, -config.GRAVITY * self.mass])
        thrust_force = self.apply_thrust(total_thrust)
        net_force = gravity_force + thrust_force

        # Angular: net torque + velocity
        angular_acceleration = torque / self.moment_of_inertia
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

        # Update position
        acceleration = net_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

    def state(self):
        """ Return the current state as a dictionary. """
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "angle": self.angle,
            "angular_velocity": self.angular_velocity
        }

    def world_to_screen(self, pos):
        """
        Convert a point in world coordinates to screen coordinates using config values.
        """
        screen_x = (pos[0] - config.CAMERA_CENTER[0]) * config.ZOOM + config.SCREEN_WIDTH / 2
        screen_y = config.SCREEN_HEIGHT / 2 - (pos[1] - config.CAMERA_CENTER[1]) * config.ZOOM
        return int(screen_x), int(screen_y)

    def draw(self, surface):
        """
        Draw the drone as a blue rectangle.
        The rectangle is defined in the drone's local coordinates (using WING_SPAN and DRONE_HEIGHT),
        rotated by its current angle and translated by its position.
        """
        # Calculate points for drone body
        w = config.WING_SPAN
        h = config.DRONE_HEIGHT
        local_points = np.array([
            [-w/2, -h/2],
            [w/2, -h/2],
            [w/2, h/2],
            [-w/2, h/2]
        ])

        # Rotate the drone to its angle
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        rotated_points = []
        for p in local_points:
            rx = p[0] * cos_a - p[1] * sin_a
            ry = p[0] * sin_a + p[1] * cos_a
            rotated_points.append((rx + self.position[0], ry + self.position[1]))

        # Translate world position to screen position
        screen_points = [self.world_to_screen(pt) for pt in rotated_points]

        # Draw the drone to the screen
        pygame.draw.polygon(surface, (0, 0, 255), screen_points)

    def draw_thrust(self, surface, left_thrust, right_thrust):
        """
        Visualize the thrust applied at each wing as red lines.
        Thrust is drawn in the direction of the applied force.
        Expect left and right thrust to be from 0 to 1
        """
        # Local positions of the wings:
        left_wing_local = np.array([-config.WING_SPAN/2, 0])
        right_wing_local = np.array([config.WING_SPAN/2, 0])
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        def local_to_world(local):
            x = local[0]*cos_a - local[1]*sin_a
            y = local[0]*sin_a + local[1]*cos_a
            return np.array([x, y]) + self.position

        left_wing_world = local_to_world(left_wing_local)
        right_wing_world = local_to_world(right_wing_local)

        # Find thrust angle and calculate end points
        thrust_dir = -np.array([math.sin(self.angle), -math.cos(self.angle)])
        vis_scale = config.WING_SPAN / 4 # Scaling factor for visualization
        left_end = left_wing_world + thrust_dir * left_thrust * vis_scale
        right_end = right_wing_world + thrust_dir * right_thrust * vis_scale

        # Draw thrusters
        pygame.draw.line(surface, (255, 0, 0), self.world_to_screen(left_wing_world), self.world_to_screen(left_end), 2)
        pygame.draw.line(surface, (255, 0, 0), self.world_to_screen(right_wing_world), self.world_to_screen(right_end), 2)
