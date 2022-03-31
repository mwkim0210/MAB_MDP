#!/usr/bin/env python3

import numpy as np
import race_parser as rp


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
                                else 1/self.action_attempts[la]

        # Q value of last action
        q_la = self._value_estimates[la]
        self._value_estimates[la] = q_la + normalizer*(reward - q_la)

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    @property
    def value_estimates(self):
        return self._value_estimates


class HorseBandit:

    def __init__(self, nA: int = 10, mode: str = "winloss"):
        """
        params
            nA (int) : Number of Action
            mode (str) : Reward mode "winloss" or "dividend"

        return None
        """
        self.mode = mode
        races = rp.load_races()
        races12 = []
        for racecode, horses in races.items():
            if len(races[racecode]) == 12:
                races12.append(horses)
        self.races = races12
        self.nA = nA
        self.counter = 0

    def reset(self):
        # shuffled_index = np.random.permutation(np.arange(12))
        # shuffled_races = []
        # for race in self.races:
        #     shuffled_races.append([race[idx] for idx in shuffled_index])
        # self.races = shuffled_races
        np.random.shuffle(self.races)
        self.counter = 0

    def pull(self, action):
        """
        params:
            race (list): races[racecode], racecode (str)
            pick (int): race index
        """
        self.counter += 1
        race = self.races[self.counter%len(self.races)]
        try:
            if int(race[action]['rcOrd']) <= 3 and len(race) > 8:
                return (10, True) if self.mode == "winloss" else (1e6*float(race[action]["rcP2Odd"]), True) # win
            elif int(race[action]['rcOrd']) <= 2 and len(race) > 7:
                return (10, True) if self.mode == "winloss" else (1e6*float(race[action]["rcP2Odd"]), True) # win
        except KeyError:
            return 0, False

        return 0, False # lose

def main():
    import policy

    hb = HorseBandit()
    a = Agent(hb, policy.RandomPolicy())
    for i in range(3):
        print(a.choose())

    print(f'{len(hb.races) =}')
    for h in hb.races[0]:
        print(h)
        break

    for i in range(6):
        print(hb.pull(a.choose()))

    print(hb.counter)
    hb.reset()
    for h in hb.races[0]:
        print(h)
    print(hb.counter)

if __name__=="__main__":
    main()

