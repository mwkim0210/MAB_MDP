from collections import defaultdict, namedtuple
import itertools

import matplotlib.pyplot as plt
import numpy as np

from env import Game


env = Game()

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def epsilon_greedy_policy(Q, epsilon, nA):
    """
    Args:
        Q: a dictionary that maps from state to action values. Each value is a array of length nA
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    Args:
        alpha: Temporal-difference learning rate (see lecture note 7 slide 3)
    """
    # Nested dictionary: state -> (action -> value)
    Q = defaultdict(lambda: np.zeros(5))
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )

    policy = epsilon_greedy_policy(Q, epsilon, 6)

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print(f"Episode {i_episode+1}")

        state = env.reset()

        for t in itertools.count():
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step_td(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


Q, stats = q_learning(env, 500)

plt.figure(1)
plt.plot(stats.episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Episode Length")
plt.title("Episode Length over Time")
plt.show()

plt.figure(2)
plt.plot(stats.episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Rewards")
plt.title("Episode Reward over Time")
plt.show()

plt.figure(3)
plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
plt.xlabel("Time Steps")
plt.ylabel("Episode")
plt.title("Episode per time step")
plt.show()

