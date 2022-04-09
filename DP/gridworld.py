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
        if shape is None:
            shape = [8, 10]
        elif not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        nS = np.prod(shape)
        nA = 4

        height = shape[0]
        width = shape[1]

        init_state = 12
        mine = [6, 8, 40]

        P = dict()
        grid = np.arange(nS).reshape(shape)

        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex  # state
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s == 58
            reward = 0.0 if is_done(s) else -1.0

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
                    ns_up = s - width if s - width not in mine else init_state
                    ns_up2 = s - width if s - width not in mine else init_state
                else:
                    ns_up = s - width if s - width not in mine else init_state
                    ns_up2 = s - 2 * width if s - 2 * width not in mine else init_state

                if y == (height - 1):
                    ns_down = s
                    ns_down2 = s
                elif y == (height - 2):
                    ns_down = s + width if s + width not in mine else init_state
                    ns_down = s + width  if s + width not in mine else init_state
                else:
                    ns_down = s + width if s + width not in mine else init_state
                    ns_down2 = s + 2 * width if s + 2 * width not in mine else init_state

                if x == 0:
                    ns_left = s
                    ns_left2 = s
                elif x == 1:
                    ns_left = s - 1 if s - 1 not in mine else init_state
                    ns_left2 = s - 1 if s - 1 not in mine else init_state
                else:
                    ns_left = s - 1 if s - 1 not in mine else init_state
                    ns_left2 = s - 2 if s - 2 not in mine else init_state

                if x == (width - 1):
                    ns_right = s
                    ns_right2 = s
                elif x == (width - 2):
                    ns_right = s + 1 if s + 1 not in mine else init_state
                    ns_right2 = s + 1 if s + 1 not in mine else init_state
                else:
                    ns_right = s + 1 if s + 1 not in mine else init_state
                    ns_right2 = s + 2 if s + 2 not in mine else init_state

                P[s][UP] = [(44/100, ns_up, reward, is_done(ns_up)), (56/100, ns_up2, reward, is_done(ns_up2))]
                P[s][RIGHT] = [(6/100, ns_right, reward, is_done(ns_right)), (94/100, ns_right2, reward, is_done(ns_right2))]
                P[s][DOWN] = [(44/100, ns_down, reward, is_done(ns_down)), (56/100, ns_down2, reward, is_done(ns_down2))]
                P[s][LEFT] = [(6/100, ns_left, reward, is_done(ns_left)), (94/100, ns_left2, reward, is_done(ns_left2))]

            it.iternext()

        # initial state distributions
        isd = np.ones(nS) / nS

        self.P = P

        super().__init__(nS, nA, P, isd)
        self.s = init_state

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

            if self.s == s:
                output = " x "
            elif s == 58:
                output = " T "
            elif s == 6 or s == 8 or s == 40:
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
