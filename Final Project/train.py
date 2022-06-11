import random
import math
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from network import DQN
from memory import ReplayMemory, Transition
from env import Game
from config import *


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
buffer = 1000
n_actions = 5
num_episodes = 50

env = Game()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(HEIGHT, WIDTH, n_actions).to(device)
target_net = DQN(HEIGHT, WIDTH, n_actions).to(device)

# load network parameters of policy network
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())

memory = ReplayMemory(buffer)
steps_done = 0


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


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # plot average of 100 episodes
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # stop to update plot


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    # Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


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

        # 선택한 action을 대입하여 reward와 done을 얻어낸다.
        # env.step(action.item())의 예시
        # >> (array([-0.008956, -0.160571,  0.005936,  0.302326]), 1.0, False, {})
        _, reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 새로운 state를 구해보자.
        last_grid = current_grid
        # action이 반영된 screen을 얻어낸다.
        current_grid = env.return_env()

        if not done:
            next_state = current_grid - last_grid
        else:  # 만약, done이 True라면 그만하자.
            next_state = None

        # 얻어낸 transition set을 memory에 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # (policy network에서) 최적화 한단계 수행
        optimize_model()

        # 마찬가지로 done이 True 라면,
        if done:
            # 하나의 episode가 몇 번 진행 되었는지 counting 하는 line
            episode_durations.append(t + 1)
            plot_durations()
            # plt.show()
            break

    # TARGET_UPDATE 마다 target network의 parameter를 update 한다.
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # (episode 한번에 대한) 전체 for문 1회 종료.

# 학습 마무리.
print('Complete')
env.render()
plt.ioff()
plt.show()


if __name__ == '__main__':
    # main()
    pass
