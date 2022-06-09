import io
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=None):
        # Your Code here
        """
        - nS: number of states
        - nA: number of actions
        - P: transitions (*)
        - isd: initial state distribution (**)

        (*) dictionary of lists, where
          P[s][a] == [(probability, next state, reward, done), ...]
        (**) list or array of length nS
        """
        x = 6
        y = 44
        id_x = x
        id_y = y
        width = 8 + x % 4
        height = 8 + y % 4
        self.width = width
        self.height = height
        init_state = ((x + y) % 3, (x - y) % 3)
        init_state = width * init_state[1] + init_state[0]

        terminal_state = (width - 1 - (x - y) % 3, height - 1 - (x + y) % 3)
        terminal_state = width * terminal_state[1] + terminal_state[0]

        # mine = [((x + y)**2 % width, (x - y)**2 % height), (x**4 % width, y ** 4 % height), ((2022 - y) % width, (2022 - x) % height)]
        mine = [(x + y) ** 2 % width + width * ((x - y) ** 2 % height), x ** 4 % width + width * (y ** 4 % height),
                (2022 - y) % width + width * ((2022 - x) % height)]

        self.init_state = init_state
        self.mine = mine
        self.terminal_state = terminal_state
        # print(self.init_state, self.mine, self.terminal_state, self.width, self.height)

        if shape is None:
            shape = [self.height, self.width]
        elif not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        nS = np.prod(shape)
        nA = 4

        P = dict()
        grid = np.arange(nS).reshape(shape)

        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex  # state
            y, x = it.multi_index
            P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s == self.terminal_state
            reward = 0.0 if is_done(s) else -1.0

            # reward_grid = 10 * np.random.rand(8, 10)

            """
            reward_grid = np.array([[ -9.59, -10.59,  -8.46, -11.46,  -7.2,  -12.2,   -5.82, -12.82,  -4.31, -13.31],
                                     [ -8.78,  -9.78,  -7.66, -10.66,  -6.4,  -11.4,   -5.02, -12.02,  -3.51, -12.51],
                                     [ -8.43,  -9.43,  -7.3,  -10.3,   -6.05, -11.05,  -4.66, -11.66,  -3.15, -12.15],
                                     [ -7.28,  -8.28,  -6.15,  -9.15,  -4.89,  -9.89,  -3.51, -10.51,  -2.,   -11.  ],
                                     [ -7.55,  -8.55,  -6.42,  -9.42,  -5.17, -10.17,  -3.78, -10.78,  -2.27, -11.27],
                                     [ -5.28,  -6.28,  -4.15,  -7.15,  -2.89,  -7.89,  -1.51,  -8.51,   0.,    -9.  ],
                                     [ -7.55,  -8.55,  -6.42,  -9.42,  -5.17, -10.17,  -3.78, -10.78,  -2.27, -11.27],
                                     [ -7.28,  -8.28,  -6.15,  -9.15,  -4.89,  -9.89,  -3.51, -10.51,  -2.,   -11.  ]])
            """
            # reward = reward_grid.reshape([nS, 1])[s]
            reward_up = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                               [0., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                               [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.]])

            reward_right = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [1., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
                                   [1., 0., 1., 0., 1., 0., 1., 0., 0., 0.],
                                   [1., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
                                   [1., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
                                   [1., 0., 1., 0., 1., 0., 1., 0., 0., 0.],
                                   [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

            reward_down = np.array([[0., 0., 1., 0., 1., 0., 1., 1., 1., 1.],
                                   [0., 0., 0., 0., 0., 0., 1., 0., 1., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                   [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
                                   [0., 1., 0., 0., 0., 0., 1., 0., 1., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

            reward_left = np.array([[0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
                                   [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                   [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                   [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                   [0., 0., 0., 1., 0., 1., 0., 1., 0., 1.],
                                   [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                   [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                                   [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.]])



            # terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                # next states
                if y == 0:
                    ns_up = s
                    ns_up2 = s
                elif y == 1:
                    ns_up = s - width if (s - width) not in mine else init_state
                    ns_up2 = s - width if (s - width) not in mine else init_state
                else:
                    ns_up = s - width if (s - width) not in mine else init_state
                    ns_up2 = s - 2 * width if (s - 2 * width) not in mine else init_state

                if y == (height - 1):
                    ns_down = s
                    ns_down2 = s
                elif y == (height - 2):
                    ns_down = s + width if (s + width) not in mine else init_state
                    ns_down2 = s + width if (s + width) not in mine else init_state
                else:
                    ns_down = s + width if (s + width) not in mine else init_state
                    ns_down2 = s + 2 * width if (s + 2 * width) not in mine else init_state

                if x == 0:
                    ns_left = s
                    ns_left2 = s
                elif x == 1:
                    ns_left = s - 1 if (s - 1) not in mine else init_state
                    ns_left2 = s - 1 if (s - 1) not in mine else init_state
                else:
                    ns_left = s - 1 if (s - 1) not in mine else init_state
                    ns_left2 = s - 2 if (s - 2) not in mine else init_state

                if x == (width - 1):
                    ns_right = s
                    ns_right2 = s
                elif x == (width - 2):
                    ns_right = s + 1 if (s + 1) not in mine else init_state
                    ns_right2 = s + 1 if (s + 1) not in mine else init_state
                else:
                    ns_right = s + 1 if (s + 1) not in mine else init_state
                    ns_right2 = s + 2 if (s + 2) not in mine else init_state
                # Modified reward version, for optimal policy at zero-discount factor
                P[s][UP] = [(id_y/100, ns_up, reward_up[s//10][s%10], is_done(ns_up)), (1-id_y/100, ns_up2, reward_up[s//10][s%10], is_done(ns_up2))]
                P[s][RIGHT] = [(id_x/100, ns_right, reward_right[s//10][s%10], is_done(ns_right)), (1-id_x/100, ns_right2, reward_right[s//10][s%10], is_done(ns_right2))]
                P[s][DOWN] = [(id_y/100, ns_down, reward_down[s//10][s%10], is_done(ns_down)), (1-id_y/100, ns_down2, reward_down[s//10][s%10], is_done(ns_down2))]
                P[s][LEFT] = [(id_x/100, ns_left, reward_left[s//10][s%10], is_done(ns_left)), (1-id_x/100, ns_left2, reward_left[s//10][s%10], is_done(ns_left2))]

                """  # Original reward version
                P[s][UP] = [(id_y/100, ns_up, reward, is_done(ns_up)), (1-id_y/100, ns_up2, reward, is_done(ns_up2))]
                P[s][RIGHT] = [(id_x/100, ns_right, reward, is_done(ns_right)), (1-id_x/100, ns_right2, reward, is_done(ns_right2))]
                P[s][DOWN] = [(id_y/100, ns_down, reward, is_done(ns_down)), (1-id_y/100, ns_down2, reward, is_done(ns_down2))]
                P[s][LEFT] = [(id_x/100, ns_left, reward, is_done(ns_left)), (1-id_x/100, ns_left2, reward, is_done(ns_left2))]
                """

            it.iternext()

        # initial state distributions
        isd = np.ones(nS) / nS

        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)
        # self.s = init_state

    def _render(self, mode='human', close=False):
        """
        Renders the current gridworld layout
        For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            #######

            if s == self.s:
                output = " x "
            elif s == self.terminal_state:
                output = " T "
            elif s in self.mine:
                output = " M "
            else:
                output = " o "

            #######

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()


def main():
    env = GridworldEnv()
    env._render()
    for k in env.P.keys():
        print(f'env.P[{k}]={env.P[k]}')


if __name__ == "__main__":
    main()
