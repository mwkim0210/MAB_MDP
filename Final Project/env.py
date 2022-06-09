import random
import copy

import numpy as np

from config import *


NEG_REWARD = -10
POS_REWARD = 10

"""
class Game(object):
    def __init__(self):
        self.grid = np.zeros((HEIGHT, LENGTH, WIDTH))
        self.location = [0, 0, 0]
        self.finished = False

    def init_game(self):
        self.grid[HEIGHT//2, :, :] = np.ones((LENGTH, WIDTH))
        self.grid[HEIGHT//2, LENGTH//2-1:LENGTH//2+1, WIDTH//2-1:WIDTH//2+1] = np.zeros((2, 2))

    def move(self, action):
        temp = self.location.copy()

        # positive direction: UP, SOUTH, EAST
        if action == 0:  # go up
            self.location[0] += 1
        elif action == 1:  # down
            self.location[0] -= 1
        elif action == 2:  # north
            self.location[1] -= 1
        elif action == 3:  # south
            self.location[1] += 1
        elif action == 4:  # east
            self.location[2] += 1
        elif action == 5:  # west
            self.location[2] -= 1
        else:
            raise ValueError(f'Incorrect action value')

        if self.location[0] < 0 or self.location[0] > HEIGHT - 1:
            self.location = temp
        if self.location[1] < 0 or self.location[1] > LENGTH - 1:
            self.location = temp
        if self.location[2] < 0 or self.location[2] > WIDTH - 1:
            self.location = temp

    def return_reward(self):
        pass
"""


class Game(object):
    def __init__(self):
        self.grid = np.zeros((HEIGHT, WIDTH))
        self.location = [0, 0]
        self.finished = False
        self.target = [HEIGHT-1, WIDTH-1]
        self.reward = 0

    def init_game(self, y=0, x=0):
        self.location = [y, x]
        self.grid[:, WIDTH//2-1:WIDTH//2+1] = np.ones(())
        self.grid[HEIGHT//2-1:HEIGHT//2+1, WIDTH//2-1:WIDTH//2] = np.zeros(())

    def move(self, action):
        # temp = self.location.copy()

        # positive direction: UP, SOUTH, EAST
        if action == 0:  # go up
            self.location[0] -= 1
        elif action == 1:  # down
            self.location[0] += 1
        elif action == 2:  # left
            self.location[1] += 1
        elif action == 3:  # right
            self.location[1] -= 1
        elif action == 4:  # flash right
            self.location[1] += 2
        else:
            raise ValueError(f'Incorrect action value')

        if self.location[0] < 0 or self.location[0] > HEIGHT - 1:
            self.finished = True
            self.reward = NEG_REWARD
        if self.location[1] < 0 or self.location[1] > WIDTH - 1:
            self.finished = True
            self.reward = NEG_REWARD
        if self.location == self.target:
            self.finished = True
            self.reward = POS_REWARD

    def return_env(self):
        env = copy.deepcopy(self.grid)
        env[self.location[0], self.location[1]] = 10
        return env

    def print_env(self):
        env = self.return_env()
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if env[i, j] == 0:
                    print(' o ', end="")
                elif env[i, j] == 1:
                    print(' = ', end="")
                else:
                    print(' x ', end="")
            print("")
        print("")


def main():
    game = Game()
    game.init_game(1, 1)
    game.print_env()
    while not game.finished:
        action = random.randint()
        game.move(i)
        game.print_env()
    game.init_game()
    # print(game.grid)


if __name__ == '__main__':
    main()
