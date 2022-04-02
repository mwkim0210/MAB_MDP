import numpy as np
from gridworld import GridworldEnv


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

    # Your code here

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
        discount_factor: gamma discount factor.

    Returns -> (policy, V):
        policy (2d numpy list): a matrix of shape [S, A] where each state s contains a valid probability distribution over actions.
        V (numpy list): V is the value function for the optimal policy.
    """
    # start with a random policy
    policy = np.zeros([env.nS, env.nA]) / env.nA

    ###

    # Your code here

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

    policy = np.zeros([env.nS, env.nA])
    V = np.zeros(env.nS)
    ###

    # Your code here

    ###
    
    return policy, V


def main():
    print('--------Policy evaluation--------')
    env = GridworldEnv()
    uniform_policy = np.ones([env.nS, env.nA])/env.nA
    v = policy_eval(uniform_policy, env)
    
    v = v.reshape(env.shape)
    print(v)

    print('---------------------------------')

    print('--------Policy iteration---------')
    
    policy, v = policy_iter(env)
    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))

    print("")
    print('---------------------------------')

    print('--------Value iteration---------')
    policy, v = value_iteration(env)

    print("Policy Probability Distribution:")
    print(policy)
    print("")

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(env.shape))
    print("")

    print('---------------------------------')


if __name__ == '__main__':
    main()