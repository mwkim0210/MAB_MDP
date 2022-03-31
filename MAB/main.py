import agent
import environment
import policy
import bandit
import datetime  # debugging

t1 = datetime.datetime.now()  # debugging

bandit = bandit.F1Bandit()
n_trials = 2500
n_experiments = 500

# This is an example.
agents = [agent.Agent(bandit, policy.RandomPolicy()),
          agent.Agent(bandit, policy.EpsilonGreedyPolicy(.01)),
          agent.Agent(bandit, policy.EpsilonGreedyPolicy(.1)),
          agent.Agent(bandit, policy.UCBPolicy(1)),
          agent.Agent(bandit, policy.UCBPolicy(2))]

env = environment.Environment(bandit, agents)
scores, optimal = env.run(n_trials, n_experiments)
env.plot_results(scores, optimal)

t2 = datetime.datetime.now()  # debugging
print(t2-t1)  # debugging
