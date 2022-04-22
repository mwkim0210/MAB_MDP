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


class ScheduledEpsilonGreedy(object):
    def __init__(self, epsilon=0.1, decay_rate=0.002):
        self.epsilon = epsilon
        self.decay_rate = decay_rate

    def __str__(self):
        return "scheduled greedy"

    def __repr__(self):
        return "scheduled \u03B5-greedy"

    def choose(self, agent):
        schedule_ver = 1
        trial_n = np.sum(agent.action_attempts)
        if schedule_ver == 1:  # exponential decay
            epsilon = self.epsilon * np.exp(- self.decay_rate * trial_n)
        elif schedule_ver == 2:  # step decay
            if trial_n < 500:
                epsilon = self.epsilon
            else:
                epsilon = self.epsilon / 10
        else:
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            best_action = np.argmax(agent.value_estimates)
            tie = np.where(agent.value_estimates == agent.value_estimates[best_action])[0]
            return np.random.choice(tie)


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

        agent.t += 1
        exploration[np.isinf(exploration)] = 10000000

        q = agent.value_estimates + exploration
        best_action = np.argmax(q)
        tie = np.where(q == q[best_action])[0]

        return np.random.choice(tie)

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def __repr__(self):
        return 'UCB (c={})'.format(self.c)


class ScheduledUCBPolicy(object):
    def __init__(self, c=1, ver=1):
        self.c = c  # initial c (degree of exploration)
        self.ver = ver

    def choose(self, agent):
        trial_n = np.sum(agent.action_attempts)
        if self.ver == 1:
            c = self.c * np.exp(- 0.002 * trial_n)
        elif self.ver == 2:
            if trial_n > 500:
                c = 3
            else:
                c = 1
        else:
            c = self.c
        exploration = np.log(agent.t + 1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / c)

        agent.t += 1
        exploration[np.isinf(exploration)] = 10000000

        q = agent.value_estimates + exploration
        best_action = np.argmax(q)
        tie = np.where(q == q[best_action])[0]

        return np.random.choice(tie)

    def __str__(self):
        return 'Scheduled UCB'

    def __repr__(self):
        return 'Scheduled UCB'


if __name__ == "__main__":
    p1 = EpsilonGreedyPolicy(1)
    p2 = RandomPolicy()
    p3 = UCBPolicy(1)
    p4 = ScheduledEpsilonGreedy()
    print(str(p4))
