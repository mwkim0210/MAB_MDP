#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import datetime


class Environment(object):
    def __init__(self, bandit, agents, label='Multi-Armed Bandit'):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=500, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        for exp in range(experiments):
            # print(f'Experiments {exp+1}/{experiments}', end='\r', flush=True)
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    if str(agent) == "f/scheduled greedy":
                        action = agent.scheduled_choose(exp)
                    else:
                        action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    scores[t, i] += reward
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments

    def plot_results(self, scores, optimal):
        plt.figure(figsize=(20, 8))
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(scores)
        plt.ylabel('Average Reward')
        plt.legend(self.agents, loc=4)
        plt.subplot(2, 1, 2)
        plt.plot(optimal * 100)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        sns.despine()
        plt.savefig('fig'+datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")+'.png')
        plt.show()

