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

RACETRACK = 'Oschersleben'  # customized


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


class PrioritizedReplayBuffer:
    def __init__(self, alpha=0.6):
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((buffer_limit,), dtype=np.float32)
        self.alpha = alpha

    def put(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < buffer_limit:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % buffer_limit

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == buffer_limit:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float)

        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        for sample in samples:
            s, a, r, s_prime, done = sample
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), \
            torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst), \
            torch.tensor(indices), weights

    def update_priorities(self, indices, prios):
        for idx, prio in zip(indices, prios):
            self.priorities[idx] = prio

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(405, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size):
        if memory_size < train_start:
            return random.randint(0, 4)
        else:
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
    # 10개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def train(q, q_target, memory, optimizer, beta=0.4):
    if memory.size() < batch_size:
        return  # 충분한 경험이 없으면 학습 건너뜀

    for _ in range(10):
        # Prioritized Experience Replay 샘플링
        s, a, r, s_prime, done_mask, indices, weights = memory.sample(batch_size, beta)

        # 다음 행동 선택은 online network(q)
        with torch.no_grad():
            next_q = q(s_prime)
            next_actions = next_q.argmax(1).unsqueeze(1)

            # 선택된 행동을 target network로 평가
            next_q_target = q_target(s_prime).gather(1, next_actions)
            target = r + gamma * next_q_target * done_mask

        # 현재 Q값
        q_out = q(s)
        q_a = q_out.gather(1, a)

        # TD-오차로 priority 업데이트
        td_error = (q_a - target).detach().squeeze().abs().cpu().numpy()
        memory.update_priorities(indices.cpu().numpy(), td_error + 1e-5)

        # 중요도 샘플링 보정 곱한 loss 계산
        loss = (F.smooth_l1_loss(q_a, target, reduction='none') * weights.unsqueeze(1)).mean()

        # 역전파 및 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def preprocess_lidar(ranges):
    eighth = int(len(ranges) / 8)

    return np.array(ranges[eighth:-eighth: 2])


def main():
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + '_' + RACETRACK)

    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)
    q = Qnet()
    # q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = PrioritizedReplayBuffer(alpha=0.6)

    # poses = np.array([[0., 0., np.radians(0)]])
    # poses = np.array([[0.8007017, -0.2753365, 4.1421595]])
    # poses = np.array([[0.8, -0.27, 4.14]])  # map_easy3
    poses = np.array([[1.5, -0.7, 0.0]])  # Oschersleben

    print_interval = 10
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    speed = 3.0
    fastlap = 10000.0
    laptimes = []
    beta_start = 0.4
    beta_frames = 100000

    for n_epi in range(10000):  # 10000 -> 3000으로 변경함
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        beta = min(1.0, beta_start + n_epi * (1.0 - beta_start) / beta_frames)
        obs, r, done, info = env.reset(poses=poses)
        s = preprocess_lidar(obs['scans'][0])
        done = False

        laptime = 0.0

        while not done:
            # env.render()

            actions = []

            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            steer = (a - 2) * (np.pi / 30)
            if a == 2:
                speed = 9.5
            elif a == 1 or a == 3:
                speed = 8.5
            else:
                speed = 8
            actions.append([steer, speed])
            actions = np.array(actions)
            obs, r, done, info = env.step(actions)
            s_prime = preprocess_lidar(obs['scans'][0])
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100, s_prime, done_mask))
            s = s_prime

            laptime += r
            # env.render(mode='human_fast')

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)
                lap_count = int(obs['lap_counts'][0])
                # print(f"[Episode {n_epi}] Lap Count: {lap_count}, Lap Time: {lap}")
                if lap_count == 2:
                    torch.save(q.state_dict(), work_dir + '_' + RACETRACK + '/fast-model' + str(
                        round(obs['lap_times'][0], 3)) + '_' + str(n_epi) + '.pt')
                    fastlap = lap
                    break

        if memory.size() > train_start:
            # train(q, q_target, memory, optimizer)
            # PER
            train(q, q_target, memory, optimizer, beta=beta)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            lap = round(obs['lap_times'][0], 3)
            lap_count = int(obs['lap_counts'][0])
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%, lap_time : {:.3f}, lap_count : {}"
                  .format(n_epi, laptime / print_interval, memory.size(), epsilon * 100, lap, lap_count))

    print('train finish')
    env.close()


if __name__ == '__main__':
    main()
    # eval()
