import os
import random
import math
from itertools import count
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from network import DQN, LargerDQN
from memory import ReplayMemory, Transition
from env import Game
from config import *


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.15
EPS_DECAY = 1000
TARGET_UPDATE = 100
buffer = 2000
n_actions = 8
num_episodes = 1000
learning_rate = 0.0003
lr_step = 2500

"""
model_path = './save/insert_file_name'
checkpoint = torch.load(model_path, device)
state_dict = checkpoint['net']
"""

env = Game()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# policy_net = DQN(HEIGHT, WIDTH, n_actions).to(device)
policy_net = LargerDQN(HEIGHT, WIDTH, n_actions).to(device)

# policy_net.load_state_dict(state_dict)

# target_net = DQN(HEIGHT, WIDTH, n_actions).to(device)
target_net = LargerDQN(HEIGHT, WIDTH, n_actions).to(device)


# load network parameters of policy network
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(policy_net.parameters(), lr=learning_rate/3)

memory = ReplayMemory(buffer)
steps_done = 0

def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'net': model.state_dict()
    }
    file_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_path = os.path.join(save_dir, file_name)
    torch.save(checkpoint, output_path)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, \
        dtype=torch.long)


episode_durations = []
reward_list = []
final_location = []


def plot_locations():
    plt.figure(1)
    plt.clf()
    locations_t = torch.tensor(final_location, dtype=torch.float)
    plt.title('Location (x)')
    plt.xlabel('Episode')
    plt.ylabel('Location (x)')
    plt.plot(locations_t.numpy())
    # plot average of 100 episodes
    if len(locations_t) >= 100:
        means = locations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # plt.pause(0.001)  # stop to update plot
    plt.savefig('./save/locations_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    return locations_t


def plot_rewards():
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(reward_list, dtype=torch.float)
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.plot(rewards_t.numpy())
    # plot average of 100 episodes
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # plt.pause(0.001)  # stop to update plot
    plt.savefig('./save/rewards_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    return rewards_t


def plot_durations():
    plt.figure(3)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Duration')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # plot average of 100 episodes
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # plt.pause(0.001)  # stop to update plot
    plt.savefig('./save/durations_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.png')
    return durations_t


def optimize_model(episode):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    # Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.from_numpy(s) for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    if episode < lr_step:
        optimizer.zero_grad()
    else:
        optimizer2.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    if episode <  lr_step:
        optimizer.step()
    else:
        optimizer2.step()


for i_episode in range(num_episodes):
    env.reset()
    last_grid = env.return_env()
    current_grid = env.return_env()
    state = current_grid - last_grid

    for t in count():
        # state shape >> torch.Size([1, 3, 40, 90])
        # action result >> tensor([[0]]) or tensor([[1]])
        state = torch.from_numpy(state)
        state = state.view(1, 1, state.shape[0], state.shape[1])
        action = select_action(state)

        # ????????? action??? ???????????? reward??? done??? ????????????.
        # env.step(action.item())??? ??????
        # >> (array([-0.008956, -0.160571,  0.005936,  0.302326]), 1.0, False, {})
        _, reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # ????????? state??? ????????????.
        last_grid = current_grid
        # action??? ????????? screen??? ????????????.
        current_grid = env.return_env()

        if not done:
            next_state = current_grid - last_grid
        else:  # ??????, done??? True?????? ????????????.
            next_state = None

        # ????????? transition set??? memory??? ??????
        memory.push(state, action, next_state, reward)

        # ?????? ????????? ??????
        state = next_state


        # ??????????????? done??? True ??????,
        if done:
            # ????????? episode??? ??? ??? ?????? ???????????? counting ?????? line
            episode_durations.append(t + 1)
            reward_list.append(reward)
            final_location.append(env.location[1])
            # plot_durations()
            # plot_rewards()
            # plot_locations()
            # plt.show()
            break

    # TARGET_UPDATE ?????? target network??? parameter??? update ??????.
    optimize_model(i_episode)
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 10 == 0:
        print(f"{i_episode:<4} {env.location=}")

    # (episode ????????? ??????) ?????? for??? 1??? ??????.

# ?????? ?????????.
print('Complete')
# env.render()
# plt.ioff()

save_dir = './save'
save_model(policy_net, save_dir)

rewards_t = plot_rewards()
durations_t = plot_durations()
locations_t = plot_locations()
print(f"{torch.mean(rewards_t[num_episodes - 5:num_episodes])=}")
print(f"{torch.mean(durations_t[num_episodes - 5:num_episodes])=}")
print(f"{torch.mean(locations_t[num_episodes - 5:num_episodes])=}")
plt.show()


if __name__ == '__main__':
    # main()
    pass
