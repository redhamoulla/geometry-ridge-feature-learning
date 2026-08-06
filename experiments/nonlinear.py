"""Track a per-observation register while a small MLP learns Friedman #1."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler


def jacobian_row(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True); y = model(x.unsqueeze(0)).squeeze(); y.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def main(seed: int = 0) -> None:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X, y = make_friedman1(n_samples=600, n_features=10, noise=1.0, random_state=seed)
    X = StandardScaler().fit_transform(X); y = (y-y.mean())/y.std()
    xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y[:,None], dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    probes = torch.tensor(X[:24], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2, weight_decay=1e-3)
    snapshots = {}
    for epoch in range(201):
        if epoch in {0, 50, 200}:
            J = torch.stack([jacobian_row(model, x) for x in probes]).detach().numpy()
            snapshots[epoch] = J @ J.T
        opt.zero_grad(); loss = torch.mean((model(xt)-yt)**2); loss.backward(); opt.step()
    def align(a,b): return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    print({"mse": float(loss), "kernel_alignment_0_200": align(snapshots[0], snapshots[200])})


if __name__ == "__main__": main()
