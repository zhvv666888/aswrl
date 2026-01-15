from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor
    d: torch.Tensor

class ReplayBuffer:
    def __init__(self, cap: int = 200000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def push(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch: int):
        return random.sample(self.buf, batch)

    def __len__(self):
        return len(self.buf)

def dqn_update(q, q_tgt, opt, batch, gamma: float):
    s = torch.stack([t.s for t in batch])
    a = torch.stack([t.a for t in batch]).long().squeeze(-1)
    r = torch.stack([t.r for t in batch]).squeeze(-1)
    s2 = torch.stack([t.s2 for t in batch])
    d = torch.stack([t.d for t in batch]).squeeze(-1)

    q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        max_q_s2 = q_tgt(s2).max(dim=1).values
        y = r + gamma * (1.0 - d) * max_q_s2

    loss = (q_sa - y).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), 10.0)
    opt.step()
    return float(loss.item())
