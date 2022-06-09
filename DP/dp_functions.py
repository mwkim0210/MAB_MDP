import numpy as np
from gridworld import GridworldEnv
import time


def policy_eval(policy, env, discount_factor=1., theta=1e-8):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
                If policy[1] == [0.1, 0, 0.9, 0], then it goes up with prob. 0.1 or goes down otherwise.
        env (GridworldEnv) : OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        discount_factor (float): Gamma discount factor.
        theta (float): We stop evaluation once our value function change is less than theta for all states.

    Returns:
        V (numpy list) : Vector of length env.nS representing the value function.
    """
    V = np.zeros(env.nS)

    ###
    # RL lecture 5 p.8

    while True:
        delta = 0
        for state in range(env.nS):
            v = V[state]
            temp = 0
            for action, action_prob in enumerate(policy[state]):
                for state_prob, next_state, reward, done in env.P[state][action]:
                    # action_prob: pi(a|s), state_prob: p(s',r|s,a)
                    temp += action_prob * state_prob * (reward + discount_factor * V[next_state])
            V[state] = temp
            delta = max(delta, np.abs(v - V[state]))

        if delta < theta:
            break

    ###
    return V


def policy_iter(env, policy_eval_fn=policy_eval, discount_factor=1.):
    """
    Policy Improvement Algorithm.
    Iteratively evaluate the policy and update it.
    Iteration terminiates if updated policy achieves optimal.

    Args:
        Args:
        env: The OpenAI environment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor:test gamma discount factor.

    Returns -> (policy, V):
        policy (2d numpy list): a matrix of shape [S, A] where each state s contains a valid probability distribution over actions.
        V (numpy list): V is the value function for the optimal policy.
    """
    # iter_flag = 0  # check number of iterations
    # start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    ###
    # RL lecture 5 p.19
    # while True:
    for i in range(1000):
        # iter_flag += 1
        policy_stable = True
        V = policy_eval_fn(policy, env, discount_factor)
        for state in range(env.nS):
            # current_action: a, policy[state]: pi(s)
            current_action = np.argmax(policy[state])
            action_values = np.zeros(env.nA)
            for action in range(env.nA):
                for state_prob, next_state, reward, done in env.P[state][action]:
                    # state_prob: p(s',r|s,a)
                    action_values[action] += state_prob * (reward + discount_factor * V[next_state])
            best_action = np.argmax(action_values)
            if current_action != best_action:
                # print(state)
                policy_stable = False
            policy[state] = np.eye(env.nA)[best_action]

        if policy_stable:
            # print(f"{iter_flag=}")
            return policy, V

    ###

    return policy, V


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    # iter_flag = 0  # check number of iterations
    policy = np.zeros([env.nS, env.nA])
    V = np.zeros(env.nS)
    ###
    # RL lecture 5 p.26
    while True:
        # iter_flag += 1
        delta = 0
        for state in range(env.nS):
            v = V[state]
            action_values = np.zeros(env.nA)
            for action in range(env.nA):
                for state_prob, next_state, reward, done in env.P[state][action]:
                    action_values[action] += state_prob * (reward + discount_factor * V[next_state])
            V[state] = np.max(action_values)

            delta = max(delta, np.abs(v - V[state]))

        if delta < theta:
            # print(f"{iter_flag=}")
            break

    # Output a deterministic policy pi
    for state in range(env.nS):
        action_values = np.zeros(env.nA)
        for action in range(env.nA):
            for state_prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += state_prob * (reward + discount_factor * V[next_state])
        best_action = np.argmax(action_values)
        policy[state, best_action] = 1.0

    ###
    
    return policy, V


def main():
    discount_factor = 0.0
    # env = GridworldEnv()

    print('--------Policy evaluation--------')
    env = GridworldEnv()
    uniform_policy = np.ones([env.nS, env.nA])/env.nA
    v = policy_eval(uniform_policy, env, discount_factor)
    
    v = v.reshape(env.shape)
    print(v)

    print('---------------------------------')

    print('--------Policy iteration---------')

    start = time.time()
    policy, v = policy_iter(env, discount_factor=discount_factor)
    policy_iter_time = time.time() - start

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    np.set_printoptions(precision=2)

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))

    print("")
    print('---------------------------------')

    print('--------Value iteration---------')

    start = time.time()
    policy, v = value_iteration(env, discount_factor=discount_factor)
    value_iter_time = time.time() - start

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    np.set_printoptions(precision=2)

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    print('---------------------------------')

    print(f"Policy Iteration Converge time: {policy_iter_time}")
    print(f"Value Iteration Converge time: {value_iter_time}")


if __name__ == '__main__':
    main()
