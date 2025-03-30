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
        """
        Return the current state as a dictionary. 
        Will need to be normalized for the neural network later.
        """
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

    def draw(self, surface, left_thrust, right_thrust):
        """
        Draw the drone as a blue rectangle and the thrust lines as red lines.
        """
        w, h = config.WING_SPAN, config.DRONE_HEIGHT
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        def local_to_world(local_point):
            return self.position + rotation_matrix @ local_point

        def draw_line(start, end, color=(255, 0, 0), width=2):
            pygame.draw.line(surface, color, self.world_to_screen(start), self.world_to_screen(end), width)

        # Drone body
        local_corners = np.array([
            [-w/2, -h/2],
            [ w/2, -h/2],
            [ w/2,  h/2],
            [-w/2,  h/2]
        ])
        world_corners = [local_to_world(p) for p in local_corners]
        screen_corners = [self.world_to_screen(p) for p in world_corners]
        pygame.draw.polygon(surface, (0, 0, 255), screen_corners)

        # Thruster lines
        wing_offset = w / 2
        thrust_dir = -np.array([math.sin(self.angle), -math.cos(self.angle)])
        scale = w / 4
        wings = {
            'left': (-wing_offset, left_thrust),
            'right': (wing_offset, right_thrust)
        }

        for _, (x_offset, thrust) in wings.items():
            wing_pos = local_to_world(np.array([x_offset, 0]))
            end_pos = wing_pos + thrust_dir * thrust * scale
            draw_line(wing_pos, end_pos)
