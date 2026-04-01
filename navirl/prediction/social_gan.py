from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as _err:
    raise ImportError(
        "PyTorch is required for SocialGAN. Install it with: pip install torch"
    ) from _err

from navirl.prediction.base import PredictionResult, TrajectoryPredictor


class _PoolingModule(nn.Module):
    """Social pooling for the GAN encoder/decoder."""

    def __init__(self, hidden_dim: int = 64, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 2, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Pool neighbouring agents using vectorized operations.

        Args:
            hidden: ``(N, hidden_dim)``
            positions: ``(N, 2)``

        Returns:
            ``(N, bottleneck_dim)``
        """
        N = hidden.size(0)

        # Vectorized version - O(N) instead of O(N²)
        # Expand positions for pairwise relative position computation
        pos_expanded = positions.unsqueeze(0)  # (1, N, 2)
        pos_broadcasted = positions.unsqueeze(1)  # (N, 1, 2)
        rel_pos = pos_expanded - pos_broadcasted  # (N, N, 2)

        # Expand hidden states to match
        hidden_expanded = hidden.unsqueeze(0).expand(N, -1, -1)  # (N, N, hidden_dim)

        # Concatenate hidden states with relative positions
        cat = torch.cat([hidden_expanded, rel_pos], dim=2)  # (N, N, hidden_dim+2)

        # Reshape for MLP processing: (N*N, hidden_dim+2)
        cat_flat = cat.view(-1, cat.size(-1))
        out_flat = self.mlp(cat_flat)  # (N*N, bottleneck_dim)

        # Reshape back and pool: (N, N, bottleneck_dim) -> (N, bottleneck_dim)
        out = out_flat.view(N, N, -1)
        pooled = torch.max(out, dim=1)[0]  # (N, bottleneck_dim)

        return pooled


class _Encoder(nn.Module):
    def __init__(self, input_dim: int = 2, embedding_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observed trajectory.

        Args:
            obs: ``(T_obs, N, 2)``

        Returns:
            (h, c) each ``(N, hidden_dim)``
        """
        T, N, _ = obs.shape
        h = torch.zeros(N, self.hidden_dim, device=obs.device)
        c = torch.zeros(N, self.hidden_dim, device=obs.device)
        for t in range(T):
            emb = torch.relu(self.embedding(obs[t]))
            h, c = self.lstm(emb, (h, c))
        return h, c


class _Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        pool_dim: int = 64,
        noise_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.lstm = nn.LSTMCell(embedding_dim + pool_dim + noise_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.pool = _PoolingModule(hidden_dim, pool_dim)
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        last_pos: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        noise: torch.Tensor,
        pred_horizon: int,
    ) -> torch.Tensor:
        """Decode future trajectory.

        Args:
            last_pos: ``(N, 2)``
            h, c: ``(N, hidden_dim)``
            noise: ``(N, noise_dim)``
            pred_horizon: Number of future steps.

        Returns:
            ``(pred_horizon, N, 2)``
        """
        preds: list[torch.Tensor] = []
        current_pos = last_pos
        for _ in range(pred_horizon):
            emb = torch.relu(self.embedding(current_pos))
            pooled = self.pool(h, current_pos)
            lstm_in = torch.cat([emb, pooled, noise], dim=1)
            h, c = self.lstm(lstm_in, (h, c))
            offset = self.output(h)
            current_pos = current_pos + offset
            preds.append(current_pos)
        return torch.stack(preds, dim=0)


class _Discriminator(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Classify a trajectory as real or fake.

        Args:
            trajectory: ``(T, N, 2)``

        Returns:
            Score per agent ``(N,)``
        """
        T, N, _ = trajectory.shape
        emb = torch.relu(self.embedding(trajectory.reshape(T * N, -1))).reshape(T, N, -1)
        _, (h, _) = self.lstm(emb)
        return self.classifier(h.squeeze(0)).squeeze(-1)


class SocialGAN(nn.Module):
    """Social GAN (Gupta et al. 2018).

    Generator: Encoder LSTM + social pooling + noise -> Decoder LSTM.
    Discriminator: classifies trajectory realism.
    Supports variety loss for diverse predictions.
    """

    def __init__(
        self,
        input_dim: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        noise_dim: int = 16,
        pool_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = _Encoder(input_dim, embedding_dim, hidden_dim)
        self.decoder = _Decoder(input_dim, embedding_dim, hidden_dim, pool_dim, noise_dim)
        self.discriminator = _Discriminator(input_dim, hidden_dim)
        self.noise_dim = noise_dim

    def sample_noise(self, N: int, device: torch.device) -> torch.Tensor:
        return torch.randn(N, self.noise_dim, device=device)

    def generate(
        self,
        obs: torch.Tensor,
        pred_horizon: int = 12,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate predicted trajectories.

        Args:
            obs: ``(T_obs, N, 2)``
            pred_horizon: Number of future steps.
            noise: Optional latent noise ``(N, noise_dim)``.

        Returns:
            ``(pred_horizon, N, 2)``
        """
        h, c = self.encoder(obs)
        N = obs.size(1)
        if noise is None:
            noise = self.sample_noise(N, obs.device)
        return self.decoder(obs[-1], h, c, noise, pred_horizon)

    def discriminate(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.discriminator(trajectory)

    @staticmethod
    def variety_loss(
        predictions: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Variety (best-of-many) loss.

        From a list of K predicted trajectories, pick the one closest to
        the ground truth and compute L2 loss on that trajectory only.

        Args:
            predictions: K tensors each of shape ``(T, N, 2)``.
            target: Ground-truth ``(T, N, 2)``.

        Returns:
            Scalar loss.
        """
        errors = []
        for pred in predictions:
            err = torch.mean(torch.norm(pred - target, dim=-1))
            errors.append(err)
        errors_tensor = torch.stack(errors)
        return errors_tensor.min()


class SocialGANPredictor(TrajectoryPredictor):
    """Inference wrapper around :class:`SocialGAN`."""

    def __init__(
        self,
        model: SocialGAN,
        horizon: int = 12,
        dt: float = 0.4,
        num_samples: int = 20,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.horizon = horizon
        self.dt = dt
        self.num_samples = num_samples
        self.device = torch.device(device)

    @torch.no_grad()
    def predict(
        self,
        observed_trajectory: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> PredictionResult:
        self.model.eval()
        context = context or {}

        obs = torch.tensor(observed_trajectory, dtype=torch.float32, device=self.device)
        obs = obs.unsqueeze(1)  # (T_obs, 1, 2)

        if "neighbor_trajectories" in context:
            neighbors = torch.tensor(
                context["neighbor_trajectories"], dtype=torch.float32, device=self.device
            )
            obs = torch.cat([obs, neighbors], dim=1)

        all_trajs: list[np.ndarray] = []
        for _ in range(self.num_samples):
            pred = self.model.generate(obs, pred_horizon=self.horizon)
            all_trajs.append(pred[:, 0, :].cpu().numpy())  # primary agent

        trajectories = np.stack(all_trajs, axis=0)  # (N_samples, T, 2)
        probabilities = np.ones(self.num_samples) / self.num_samples
        timestamps = np.arange(1, self.horizon + 1) * self.dt

        return PredictionResult(
            trajectories=trajectories,
            probabilities=probabilities,
            timestamps=timestamps,
        )
