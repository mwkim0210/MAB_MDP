import agent
import environment
import policy
import bandit

bandit = bandit.F1Bandit(reward_policy_ver=2)
n_trials = 1000
n_experiments = 1000

p1 = policy.RandomPolicy()
p2 = policy.EpsilonGreedyPolicy(.01)
p3 = policy.EpsilonGreedyPolicy(.1)
p4 = policy.UCBPolicy(1)
p5 = policy.UCBPolicy(2)
p6 = policy.ScheduledEpsilonGreedy()
p7 = policy.ScheduledUCBPolicy(c=2, ver=2)

policy_list = [p1, p2, p3, p7, p5]

agents = [agent.Agent(bandit, policy_model) for policy_model in policy_list]
"""
agents = [agent.Agent(bandit, p1),
          agent.Agent(bandit, p2),
          agent.Agent(bandit, p3),
          agent.Agent(bandit, p4),
          agent.Agent(bandit, p5),
          agent.Agent(bandit, p6)]
"""
env = environment.Environment(bandit, agents)
scores, optimal = env.run(n_trials, n_experiments)

# print results
print(f"{'idx':^3}", end="")
for policy in policy_list:
    print(f"{str(policy):^18}", end="")
print("")
i = 0
length = len(optimal)
policy_num = len(policy_list)
for i in range(0, length, 100):
    print(f"{i:^03}", end="")
    for j in range(0, policy_num):
        print(f"{optimal[i][j]:^18.3f}", end="")
        # print(f"{i:^4}{optimal[i][0]:^18}{optimal[i][1]:^18}{optimal[i][2]:^18}{optimal[i][3]:^18}{optimal[i][4]:^18}")
    print("")

print(f"{length-1:03d}", end="")
for j in range(0, policy_num):
    print(f"{optimal[length-1][j]:^18.3f}", end="")


env.plot_results(scores, optimal)
