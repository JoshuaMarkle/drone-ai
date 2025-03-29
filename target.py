import numpy as np
import pygame
import config

class Target:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

    def world_to_screen(self, pos):
        screen_x = (pos[0] - config.CAMERA_CENTER[0]) * config.ZOOM + config.SCREEN_WIDTH / 2
        screen_y = config.SCREEN_HEIGHT / 2 - (pos[1] - config.CAMERA_CENTER[1]) * config.ZOOM
        return int(screen_x), int(screen_y)

    def draw(self, surface):
        """ Draw the target as a red circle. """
        target_screen = self.world_to_screen(self.position)
        pygame.draw.circle(surface, (255, 0, 0), target_screen, 5)
