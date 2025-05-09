import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 0.001
        self.gamma = 0.99
        self.batch_size = 128
        self.buffer_size = 10000
        self.target_update = 500
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 60000
        self.n_steps = 3
        self.alpha = 0.6
        self.beta_start = 0.4
        self.beta_end = 1.0
        self.beta_steps = 50000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = SumTree(self.buffer_size)
        self.step_count = 0
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.95)
        self.memory_counter = 0
        
    def select_action(self, state, epsilon):
        self.step_count += 1
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:]) if self.memory.n_entries > 0 else 1.0
        self.memory.add(max_priority, experience)
        self.memory_counter += 1

    def sample_batch(self, beta):
        batch = []
        idxs = []
        priorities = []
        segment = self.memory.total() / self.batch_size
        for i in range(self.batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.memory.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        sampling_probabilities = np.array(priorities) / self.memory.total()
        weights = (self.memory.n_entries * sampling_probabilities) ** (-beta)
        weights /= weights.max()
        return batch, idxs, torch.FloatTensor(weights).to(self.device)

    def compute_n_step_target(self, batch, idxs):
        batch_size = len(batch)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
    
        targets = rewards.clone()
        for i in range(1, self.n_steps):
            future_idxs = [idx + i for idx in idxs if idx + i < self.memory_counter]
            if future_idxs:
                future_batch = []
                valid_future_idxs = []
                for idx in future_idxs:
                    if idx < self.memory.n_entries + self.memory.capacity - 1:
                        future_batch.append(self.memory.data[idx % self.memory.capacity])
                        valid_future_idxs.append(idx)
                
                if len(future_batch) > 0:
                    future_rewards = torch.FloatTensor([e.reward for e in future_batch]).to(self.device)
                    future_dones = torch.FloatTensor([e.done for e in future_batch]).to(self.device)
                    mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                    mask[:len(valid_future_idxs)] = True
                    targets[mask] += (self.gamma ** i) * future_rewards[:len(valid_future_idxs)] * (1 - future_dones[:len(valid_future_idxs)])
    
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            target_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets += (self.gamma ** self.n_steps) * target_q * (1 - dones)
    
        return states, actions, targets

    def compute_loss(self, batch, idxs, weights):
        if not batch:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        states, actions, targets = self.compute_n_step_target(batch, idxs)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = (q_values.detach() - targets).cpu().numpy()
        loss = (weights * (q_values - targets) ** 2).mean()
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + 0.01) ** self.alpha
            self.memory.update(idx, priority)
        return loss

    def update(self):
        if self.memory.n_entries < self.batch_size + self.n_steps - 1:
            return
        beta = self.beta_start + (self.beta_end - self.beta_start) * min(1.0, self.step_count / self.beta_steps)
        batch, idxs, weights = self.sample_batch(beta)
        loss = self.compute_loss(batch, idxs, weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())