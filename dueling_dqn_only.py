import os
import sys
import collections
import random
import gym
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Hyperparameters
learning_rate = 0.00005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
train_start = 20000

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

RACETRACK = 'Oschersleben'


def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class DuelingQnet(nn.Module):
    def __init__(self):
        super(DuelingQnet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(405, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [405] → [1, 405]
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


    def sample_action(self, obs, epsilon, memory_size):
        if memory_size < train_start:
            return random.randint(0, 4)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 4)
        else:
            return out.argmax().item()

    def action(self, obs):
        out = self.forward(obs)
        return out.argmax().item()


def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def preprocess_lidar(ranges):
    eighth = int(len(ranges) / 8)
    return np.array(ranges[eighth:-eighth:2])


def main():
    today = get_today()
    work_dir = "./" + today + '_' + RACETRACK
    os.makedirs(work_dir)

    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = DuelingQnet()
    q_target = DuelingQnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    poses = np.array([[1.5, -0.7, 0.0]])  # Oschersleben
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    print_interval = 10
    laptimes = []
    fastlap = 10000.0

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        obs, r, done, info = env.reset(poses=poses)
        s = preprocess_lidar(obs['scans'][0])
        done = False
        laptime = 0.0

        while not done:
            actions = []
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            steer = (a - 2) * (np.pi / 30)
            if a == 2:
                speed = 8.5
            elif a == 1 or a == 3:
                speed = 8.0
            else:
                speed = 7.5
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            s_prime = preprocess_lidar(obs['scans'][0])
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100, s_prime, done_mask))
            s = s_prime
            laptime += r

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)
                lap_count = int(obs['lap_counts'][0])
                if lap_count == 2 and fastlap > lap:
                    torch.save(q.state_dict(), os.path.join(work_dir, f"du-fast-model{lap}_{n_epi}.pt"))
                    fastlap = lap
                    break

        if memory.size() > train_start:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            lap = round(obs['lap_times'][0], 3)
            lap_count = int(obs['lap_counts'][0])
            print(f"n_episode :{n_epi}, score : {laptime / print_interval:.1f}, "
                  f"n_buffer : {memory.size()}, eps : {epsilon * 100:.1f}%, "
                  f"lap_time : {lap:.3f}, lap_count : {lap_count}")

    print('train finish')
    env.close()


def eval():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    f1_root = os.path.abspath(os.path.join(current_dir, "../../../"))

    env = gym.make('f110_gym:f110-v0',
                   map=os.path.join(current_dir, "maps", RACETRACK),
                   map_ext=".png", num_agents=1)

    q = DuelingQnet()
    model_path = os.path.join(f1_root, "2025-06-15_16-05-29_Oschersleben", "du-fast-model69.8_3287.pt")
    q.load_state_dict(torch.load(model_path))

    poses = np.array([[1.5, -0.7, 0.0]])
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        s = preprocess_lidar(obs['scans'][0])
        env.render()
        done = False
        laptime = 0.0

        while not done:
            actions = []
            a = q.action(torch.from_numpy(s).float())
            steer = (a - 2) * (np.pi / 30)
            if a == 2:
                speed = 8.5
            elif a in [1, 3]:
                speed = 8.0
            else:
                speed = 7.5
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            s_prime = preprocess_lidar(obs['scans'][0])
            s = s_prime
            laptime += r
            env.render(mode='human_fast')

            if done:
                lap = round(obs['lap_times'][0], 3)
                lap_count = int(obs['lap_counts'][0])
                print(f"[Eval Run {t}] Lap Count: {lap_count}, Lap Time: {lap}")
                break

    env.close()


if __name__ == '__main__':
    #main()
    eval()