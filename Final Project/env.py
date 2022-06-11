import random
import copy

import numpy as np

from config import *


NEG_REWARD = -10
POS_REWARD = 10


class Game(object):
    def __init__(self):
        self.grid = np.zeros((HEIGHT, WIDTH))
        self.location = [0, 0]
        self.finished = False
        self.target = [HEIGHT-1, WIDTH-1]
        self.reward = 0
        self.num_actions = 0

    def reset(self):
        self.finished = False
        self.reward = 0
        self.num_actions = 0

        y = random.randint(0, HEIGHT-1)
        x = random.randint(0, WIDTH//2-2)
        # x, y = 1, 1
        self.location = [y, x]
        self.grid = np.zeros((HEIGHT, WIDTH))
        self.grid[:, WIDTH//2-1:WIDTH//2+1] = np.ones(())
        self.grid[HEIGHT//2-2:HEIGHT//2+2, WIDTH//2-1:WIDTH//2] = np.zeros(())

    def step(self, action):
        self.num_actions += 1
        temp = self.location.copy()

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
            self.location = temp
        if self.location[1] < 0 or self.location[1] > WIDTH - 1:
            self.finished = True
            self.location = temp
        if self.location[1] > WIDTH - 3:  # arrived at target
            self.finished = True
        if self.finished is False:
            if self.grid[self.location[0], self.location[1]] == 1:
                self.finished = True
                self.location = temp

        if self.finished:
            self.reward = self.reward_policy()

        return self.return_env(), self.reward, self.finished

    def reward_policy(self):
        reward = 0
        reward += self.location[1]
        if self.location[1] > WIDTH//2:
            reward += self.location[1]

        if self.grid[self.location[0], self.location[1]] == 1:
            reward -= 100
        reward -= self.num_actions
        return reward

    def return_env(self):
        env = copy.deepcopy(self.grid)
        env[self.location[0], self.location[1]] = 10
        return env

    def render(self):
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

    def _print_reward_policy(self):
        reward = 0
        self.finished = True
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.reward = 0
                self.location = [i, j]
                reward = self.reward_policy()
                print(f"{reward:>4}", end="")
            print("")


def main():
    game = Game()
    game.reset()
    game.render()
    """
    while not game.finished:
        game.render()
        action = random.randint(0, 4)
        game.step(action)
    """
    game.reset()
    game._print_reward_policy()

    # print(game.grid)


if __name__ == '__main__':
    main()
