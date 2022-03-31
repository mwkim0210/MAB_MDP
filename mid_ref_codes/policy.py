#!/usr/bin/env python3

import numpy as np


class EpsilonGreedyPolicy:

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            best_action = np.argmax(agent.value_estimates)
            tie = np.where(agent.value_estimates == agent.value_estimates[best_action])[0]
            return np.random.choice(tie)

    def __str__(self):
        return f'\u03B5-greedy (\u03B5={self.epsilon})'

    def __repr__(self):
        return f'\u03B5-greedy (\u03B5={self.epsilon})'


class GreedyPolicy(EpsilonGreedyPolicy):
    """
    The Greedy policy only takes the best apparent action, with ties broken by
    random selection. This can be seen as a special case of EpsilonGreedy where
    epsilon = 0 i.e. always exploit.
    """
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return 'greedy'

    def __repr__(self):
        return 'greedy'


class RandomPolicy(EpsilonGreedyPolicy):

    def __init__(self):
        super(RandomPolicy, self).__init__(1)

    def __str__(self):
        return "random"

    def __repr__(self):
        return "random"


class UCBPolicy:
    def __init__(self, c):
        self.c = c

    def choose(self, agent):
        exploration = np.log(agent.t+1)/agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.c)

        q = agent.value_estimates + exploration
        best_action = np.argmax(q)
        tie = np.where(q == q[best_action])[0]

        return np.random.choice(tie)

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def __repr__(self):
        return 'UCB (c={})'.format(self.c)


if __name__=="__main__":
    p = EpsilonGreedyPolicy(1)
    p = RandomPolicy()
    p = UCBPolicy(1)
