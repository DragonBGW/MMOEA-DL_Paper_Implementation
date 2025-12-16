import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm
from pointer_net import PointerNet
from utils import gen_tasks, compute_tour_length
import torch.nn as nn
import os

# ------------------------
# Dataset
# ------------------------
class TSPDataset(Dataset):
    def __init__(self, n_samples, n_nodes):
        self.n = n_nodes
        self.data = [gen_tasks(n_nodes, seed=i) for i in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].astype(np.float32)

# ------------------------
# Critic Model
# ------------------------
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

# ------------------------
# Reward
# ------------------------
def reward_from_seq(seq, points, depot):
    tour = seq.cpu().numpy().tolist()
    return -compute_tour_length(tour, points.numpy(), depot.numpy())

# ------------------------
# Training Function
# ------------------------
def train(actor, critic, n_nodes=20, device='cpu', epochs=10, batch=128, n_samples=20000, lr=5e-4, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    dataset = TSPDataset(n_samples, n_nodes)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)

    actor.to(device)
    critic.to(device)

    opt_a = optim.Adam(actor.parameters(), lr=lr)
    opt_c = optim.Adam(critic.parameters(), lr=lr)

    depot = torch.zeros(1, 2).to(device)

    for ep in range(epochs):
        pbar = tqdm.tqdm(loader, desc=f"Nodes={n_nodes} | Epoch {ep+1}/{epochs}")
        for pts in pbar:
            pts = pts.to(device)
            batch_sz = pts.size(0)

            # Actor forward pass
            seqs, logps = actor(pts)
            rewards = torch.tensor(
                [reward_from_seq(seqs[i], pts[i], depot[0]) for i in range(batch_sz)],
                dtype=torch.float32, device=device
            )

            # Critic baseline
            with torch.no_grad():
                baseline_in = pts.mean(dim=1)
            baseline = critic(baseline_in).squeeze()

            # Actor loss
            advantage = rewards - baseline
            loss_a = -(advantage.detach() * logps).mean()
            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()

            # Critic loss
            pred = critic(baseline_in).squeeze()
            loss_c = F.mse_loss(pred, rewards)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

            pbar.set_postfix({"ActorLoss": loss_a.item(), "CriticLoss": loss_c.item()})

    # Save separate models for each n_nodes
    actor_path = os.path.join(save_dir, f"actor_pointernet_{n_nodes}.pt")
    critic_path = os.path.join(save_dir, f"critic_pointernet_{n_nodes}.pt")
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)

    print(f"[INFO] Saved actor model for n_nodes={n_nodes} to {actor_path}")
    print(f"[INFO] Saved critic model for n_nodes={n_nodes} to {critic_path}")

    return actor, critic

# ------------------------
# Train for multiple n_nodes
# ------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for n_nodes in [20, 30, 40, 50]: #random generate n_nodes = (20,50)
        actor = PointerNet()
        critic = Critic()
        train(actor, critic, n_nodes=n_nodes, device=device, epochs=20, batch=128, n_samples=20000)
