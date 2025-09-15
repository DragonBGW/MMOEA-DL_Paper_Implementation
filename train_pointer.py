# train_pointer.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pointer_net import PointerNet
from utils import gen_tasks, compute_tour_length
import tqdm

class TSPDataset(Dataset):
    def __init__(self, n_samples, n_nodes):
        self.n = n_nodes
        self.data = [gen_tasks(n_nodes, seed=i) for i in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32)

def reward_from_seq(seq, points, depot):
    # seq: seq length n (indices), points: n x 2
    tour = seq.cpu().numpy().tolist()
    return -compute_tour_length(tour, points.numpy(), depot.numpy())  # negative cost as reward

def train(actor, critic, device='cpu', epochs=10, batch=64, n_nodes=20, n_samples=20000, lr=5e-4):
    dataset = TSPDataset(n_samples, n_nodes)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    actor.to(device)
    critic.to(device)
    opt_a = optim.Adam(actor.parameters(), lr=lr)
    opt_c = optim.Adam(critic.parameters(), lr=lr)
    depot = torch.zeros(1,2).to(device)  # origin at (0,0) for simplicity
    for ep in range(epochs):
        pbar = tqdm.tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        epoch_loss_a = 0.0
        epoch_loss_c = 0.0
        for pts in pbar:
            pts = pts.to(device)
            batch_sz = pts.size(0)
            seqs, logps = actor(pts)  # greedy sampling inside actor
            # compute rewards per sample
            rewards = []
            for i in range(batch_sz):
                r = reward_from_seq(seqs[i], pts[i], depot[0])
                rewards.append(r)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            # baseline from critic: pass encoded features (mean)
            with torch.no_grad():
                baseline_in = pts.mean(dim=1)  # batch x 2
            baseline = critic(baseline_in).squeeze()  # batch
            advantage = rewards - baseline
            # actor loss: -adv * logp
            loss_a = -(advantage.detach() * logps).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
            # critic loss: MSE to rewards
            pred = critic(baseline_in).squeeze()
            loss_c = F.mse_loss(pred, rewards)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()
            epoch_loss_a += loss_a.item()
            epoch_loss_c += loss_c.item()
            pbar.set_postfix({"La": epoch_loss_a/(pbar.n+1), "Lc": epoch_loss_c/(pbar.n+1)})
    return actor, critic

# Critic model
import torch.nn as nn
class Critic(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)
