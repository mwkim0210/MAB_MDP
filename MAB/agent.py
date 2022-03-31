#!/usr/bin/env python3
import numpy as np


class Agent:
    def __init__(self, bandit, policy, gamma=None):
        self.policy = policy
        self.nA = bandit.nA
        self._value_estimates = np.zeros(self.nA)
        self.action_attempts = np.zeros(self.nA)
        self.t = 0
        self.last_action = None
        self.gamma = gamma

    def reset(self):
        self._value_estimates[:] = 0
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        # define "la" to shorten code
        la = self.last_action
        self.action_attempts[la] += 1

        normalizer = self.gamma if self.gamma is not None \
            else 1 / self.action_attempts[la]

        # Q value of last action
        q_la = self._value_estimates[la]
        self._value_estimates[la] = q_la + normalizer * (reward - q_la)

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    @property
    def value_estimates(self):
        return self._value_estimates


def main():
    pass


if __name__ == "__main__":
    main()
