import numpy as np

import race_loader
from config import *


class F1Bandit(object):
    def __init__(self, nA=16, reward_policy_ver=2):
        """
        :param nA: Number of actions, i.e., number of drivers to choose from
        """
        self.nA = nA
        self.reward_policy = reward_policy_ver
        self.counter = 0
        self.data = race_loader.load_race_data()  # self.data contains all Grand Prix infos(FPs, Q, Race)
        self.drivers = race_loader.load_drivers()
        self.data_list = list(self.data)

    def reset(self):
        self.counter = 0
        np.random.shuffle(self.data_list)

    def pull(self, action):
        """
        Args:
            action (int): The action you choose
        Returns:
            reward (float): The reward you design
            is_optimal (bool): True (if you win) or False (otherwise).
        """
        session_list = ['FP1', 'FP2', 'FP3', 'Q', 'Race']
        if self.reward_policy == 1:
            gp_idx = self.counter // 5
            session_idx = self.counter % 5
            session = self.data[self.data_list[gp_idx % len(self.data)]][session_list[session_idx]]
            result = 0
            for i in range(self.nA):
                if self.drivers[str(action)]['name'] == session[i]['name']:
                    result = i + 1
                    break
            self.counter += 1

            if THRESHOLD >= result > 0:
                return 10, True
            return 0, False

        elif self.reward_policy == 2:
            point_list = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
            # data_list = list(self.data)
            gp_idx = self.counter // 5
            session_idx = self.counter % 5
            session = self.data[self.data_list[gp_idx % len(self.data)]][session_list[session_idx]]
            result = 0
            for i in range(self.nA):
                if self.drivers[str(action)]['name'] == session[i]['name']:
                    result = i + 1
                    break
            self.counter += 1

            reward = point_list[result-1] if result <= 10 else 0

            if THRESHOLD >= result > 0:
                return reward, True
            return reward, False
        # return reward, is_optimal


def main():
    from pprint import PrettyPrinter
    pprinter = PrettyPrinter()
    f1 = F1Bandit(reward_policy_ver=1)
    # pprinter.pprint(f1.drivers)
    # for i, (key, value) in enumerate(f1.drivers.items()):
    #    print(i, key, value['name'])
    # pprinter.pprint(f1.data['Bahrain Grand Prix']['FP1'])
    # for i in range(16):
    #     print(f1.data['Bahrain Grand Prix']['FP1'][i]['name'])
    for i in range(16):
        reward, result = f1.pull(i)
        print(reward, result)


if __name__ == '__main__':
    main()
