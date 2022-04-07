import agent
import environment
import policy
import bandit

bandit = bandit.F1Bandit(reward_policy_ver=3)
n_trials = 2500
n_experiments = 2500

p1 = policy.RandomPolicy()
p2 = policy.EpsilonGreedyPolicy(.01)
p3 = policy.EpsilonGreedyPolicy(.1)
p4 = policy.UCBPolicy(1)
p5 = policy.UCBPolicy(2)

agents = [agent.Agent(bandit, p1),
          agent.Agent(bandit, p2),
          agent.Agent(bandit, p3),
          agent.Agent(bandit, p4),
          agent.Agent(bandit, p5)]

env = environment.Environment(bandit, agents)
scores, optimal = env.run(n_trials, n_experiments)

print(f"{'idx':^4}{str(p1):^20}{str(p2):^20}{str(p3):^20}{str(p4):^20}{str(p5):^20}")
length = len(optimal)
i = 0
for i in range(0, length, 100):
    print(f"{i:^4}{optimal[i][0]:^20}{optimal[i][1]:^20}{optimal[i][2]:^20}{optimal[i][3]:^20}{optimal[i][4]:^20}")
print(f"{length-1:^4}{optimal[length-1][0]:^20}{optimal[length-1][1]:^20}{optimal[length-1][2]:^20}{optimal[length-1][3]:^20}{optimal[length-1][4]:^20}")

env.plot_results(scores, optimal)
