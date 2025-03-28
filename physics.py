# physics.py
import numpy as np
import math
from config import MAX_THRUST, WING_SPAN, GRAVITY

class Drone:
    """
    A simple drone model in 2D space with position, velocity, and orientation.
    Thrust is applied along the drone's local downward axis.
    """

    def __init__(self, position, angle=0.0, mass=1.0, moment_of_inertia=1.0):
        """
        Initialize the drone.
        
        Args:
            position (iterable): Initial (x, y) position.
            angle (float): Initial orientation in radians.
            mass (float): Mass of the drone.
            moment_of_inertia (float): Rotational inertia of the drone.
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.angle = angle
        self.angular_velocity = 0.0
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia

    def apply_thrust(self, thrust):
        """
        Compute the thrust force vector in world coordinates.
        Thrust is applied along the drone's local downward axis (i.e. (0, -1) in local coordinates).
        
        Args:
            thrust (float): Total thrust magnitude (from both wings combined).
            
        Returns:
            np.ndarray: Thrust force vector [Fx, Fy] in world coordinates.
        """
        # Reversed thrust direction.
        fx = -thrust * math.sin(self.angle)
        fy = thrust * math.cos(self.angle)
        return np.array([fx, fy])

    def update(self, dt, left_thrust_cmd=0.0, right_thrust_cmd=0.0):
        """
        Update the drone's state over a timestep dt using left and right wing thrust commands.
        Each command is clamped by (0, MAX_THRUST). The total thrust is applied along the local
        downward direction and the difference generates a torque.
        
        Args:
            dt (float): Time step in seconds.
            left_thrust_cmd (float): Command for left wing thrust.
            right_thrust_cmd (float): Command for right wing thrust.
        """

        # Clamp thrust values
        left_thrust = max(min(left_thrust_cmd, MAX_THRUST), 0)
        right_thrust = max(min(right_thrust_cmd, MAX_THRUST), 0)
        total_thrust = left_thrust + right_thrust

        # Reversed torque direction: differential thrust * (half the wing span), multiplied by -1.
        torque = -(left_thrust - right_thrust) * (WING_SPAN / 2)
        
        gravity_force = np.array([0, -GRAVITY * self.mass])
        thrust_force = self.apply_thrust(total_thrust)
        net_force = gravity_force + thrust_force
        
        # Update linear dynamics.
        acceleration = net_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Update rotational dynamics.
        angular_acceleration = torque / self.moment_of_inertia
        self.angular_velocity += angular_acceleration * dt
        self.angle += self.angular_velocity * dt

    def state(self):
        """
        Returns the current state of the drone as a dictionary
        """

        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "angle": self.angle,
            "angular_velocity": self.angular_velocity
        }

class Target:
    """
    A simple target object that holds a fixed position.
    """

    def __init__(self, position):
        self.position = np.array(position, dtype=float)
