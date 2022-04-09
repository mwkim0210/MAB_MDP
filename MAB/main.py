import agent
import environment
import policy
import bandit

bandit = bandit.F1Bandit(reward_policy_ver=3)
n_trials = 10
n_experiments = 500

p1 = policy.RandomPolicy()
p2 = policy.EpsilonGreedyPolicy(.01)
p3 = policy.EpsilonGreedyPolicy(.1)
p4 = policy.UCBPolicy(1)
p5 = policy.UCBPolicy(2)
p6 = policy.ScheduledEpsilonGreedy()

agents = [agent.Agent(bandit, p1),
          agent.Agent(bandit, p2),
          agent.Agent(bandit, p3),
          agent.Agent(bandit, p4),
          agent.Agent(bandit, p5),
          agent.Agent(bandit, p6)]

env = environment.Environment(bandit, agents)
scores, optimal = env.run(n_trials, n_experiments)

print(f"{'idx':^4}{str(p1):^18}{str(p2):^18}{str(p3):^18}{str(p4):^18}{str(p5):^18}{str(p6):^18}")
length = len(optimal)
i = 0
for i in range(0, length, 100):
    print(f"{i:^4}{optimal[i][0]:^18}{optimal[i][1]:^18}{optimal[i][2]:^18}{optimal[i][3]:^18}{optimal[i][4]:^18}{optimal[i][5]:^18}")
print(f"{length-1:^4}{optimal[length-1][0]:^18}{optimal[length-1][1]:^18}{optimal[length-1][2]:^18}{optimal[length-1][3]:^18}{optimal[length-1][4]:^18}{optimal[length-1][5]:^18}")

env.plot_results(scores, optimal)
