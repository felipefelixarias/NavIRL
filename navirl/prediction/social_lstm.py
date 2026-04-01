from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as _err:
    raise ImportError(
        "PyTorch is required for SocialLSTM. Install it with: pip install torch"
    ) from _err

from navirl.prediction.base import PredictionResult, TrajectoryPredictor


class SocialPooling(nn.Module):
    """Grid-based social pooling layer (Alahi et al. 2016).

    Aggregates the hidden states of neighbouring agents into a fixed-size
    tensor by dividing the local neighbourhood into a spatial grid.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        grid_size: int = 4,
        neighborhood_size: float = 2.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size

        self.embedding = nn.Linear(hidden_dim * grid_size * grid_size, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        agent_idx: int,
    ) -> torch.Tensor:
        """Pool hidden states of neighbours for *agent_idx*.

        Args:
            hidden_states: ``(N_agents, hidden_dim)``
            positions: ``(N_agents, 2)``
            agent_idx: Index of the target agent.

        Returns:
            Pooled tensor of shape ``(hidden_dim,)``.
        """
        N = hidden_states.size(0)
        grid = torch.zeros(
            self.grid_size, self.grid_size, self.hidden_dim, device=hidden_states.device
        )

        ref_pos = positions[agent_idx]
        for j in range(N):
            if j == agent_idx:
                continue
            rel = positions[j] - ref_pos
            # Check neighbourhood bounds.
            if (
                torch.abs(rel[0]) > self.neighborhood_size
                or torch.abs(rel[1]) > self.neighborhood_size
            ):
                continue

            # Map relative position to grid cell.
            cell_x = int(
                ((rel[0] + self.neighborhood_size) / (2 * self.neighborhood_size) * self.grid_size)
                .clamp(0, self.grid_size - 1)
                .item()
            )
            cell_y = int(
                ((rel[1] + self.neighborhood_size) / (2 * self.neighborhood_size) * self.grid_size)
                .clamp(0, self.grid_size - 1)
                .item()
            )
            grid[cell_x, cell_y] += hidden_states[j]

        pooled = grid.view(-1)  # (grid_size * grid_size * hidden_dim)
        return self.embedding(pooled)


class SocialLSTM(nn.Module):
    """Social LSTM model (Alahi et al. 2016).

    Each agent is modelled by its own LSTM; a social pooling mechanism
    captures interactions between neighbouring agents.  The output at
    each time-step parameterises a bivariate Gaussian distribution over
    the next position.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        grid_size: int = 4,
        neighborhood_size: float = 2.0,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_embedding = nn.Linear(input_dim, embedding_dim)
        self.social_pooling = SocialPooling(hidden_dim, grid_size, neighborhood_size)
        self.lstm = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 5)  # mu_x, mu_y, sigma_x, sigma_y, rho
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h, c

    def forward(
        self,
        obs_trajectories: torch.Tensor,
        pred_horizon: int = 12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs_trajectories: Observed trajectories, shape ``(T_obs, N_agents, 2)``.
            pred_horizon: Number of future steps to predict.

        Returns:
            Tuple of predicted means ``(pred_horizon, N_agents, 2)`` and
            Gaussian parameters ``(pred_horizon, N_agents, 5)``.
        """
        T_obs, N, _ = obs_trajectories.shape
        device = obs_trajectories.device

        h, c = self.init_hidden(N, device)

        # ---------- Observation phase ----------
        for t in range(T_obs):
            positions = obs_trajectories[t]  # (N, 2)
            embedded = self.dropout(torch.relu(self.input_embedding(positions)))  # (N, emb)

            # Social pooling for each agent.
            social_tensors: list[torch.Tensor] = []
            for i in range(N):
                pooled = self.social_pooling(h, positions, i)
                social_tensors.append(pooled)
            social = torch.stack(social_tensors, dim=0)  # (N, hidden)

            lstm_input = torch.cat([embedded, social], dim=1)  # (N, emb+hidden)
            h, c = self.lstm(lstm_input, (h, c))

        # ---------- Prediction phase ----------
        pred_means = []
        gaussian_params_list = []
        current_pos = obs_trajectories[-1]  # (N, 2)

        for _ in range(pred_horizon):
            embedded = self.dropout(torch.relu(self.input_embedding(current_pos)))

            social_tensors = []
            for i in range(N):
                pooled = self.social_pooling(h, current_pos, i)
                social_tensors.append(pooled)
            social = torch.stack(social_tensors, dim=0)

            lstm_input = torch.cat([embedded, social], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            gaussian_params = self.output_layer(h)  # (N, 5)
            mu = gaussian_params[:, :2]
            # Ensure positive sigmas.
            gaussian_params = torch.cat(
                [mu, torch.exp(gaussian_params[:, 2:4]), torch.tanh(gaussian_params[:, 4:5])],
                dim=1,
            )

            current_pos = mu
            pred_means.append(mu)
            gaussian_params_list.append(gaussian_params)

        pred_means_tensor = torch.stack(pred_means, dim=0)  # (T_pred, N, 2)
        gaussian_params_tensor = torch.stack(gaussian_params_list, dim=0)  # (T_pred, N, 5)

        return pred_means_tensor, gaussian_params_tensor

    @staticmethod
    def bivariate_gaussian_nll(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood of a bivariate Gaussian.

        Args:
            predictions: Not used directly (means are embedded in *params*).
            targets: Ground-truth positions ``(T, N, 2)``.
            params: Gaussian parameters ``(T, N, 5)`` —
                ``[mu_x, mu_y, sigma_x, sigma_y, rho]``.

        Returns:
            Scalar NLL loss.
        """
        mu_x = params[:, :, 0]
        mu_y = params[:, :, 1]
        sigma_x = params[:, :, 2]
        sigma_y = params[:, :, 3]
        rho = params[:, :, 4]

        dx = targets[:, :, 0] - mu_x
        dy = targets[:, :, 1] - mu_y

        z = (
            (dx / sigma_x) ** 2
            + (dy / sigma_y) ** 2
            - 2 * rho * dx * dy / (sigma_x * sigma_y)
        )
        denom = 1.0 - rho**2 + 1e-6
        nll = (
            0.5 * z / denom
            + torch.log(sigma_x)
            + torch.log(sigma_y)
            + 0.5 * torch.log(1 - rho**2 + 1e-6)
            + np.log(2 * np.pi)
        )
        return nll.mean()


class SocialLSTMPredictor(TrajectoryPredictor):
    """Inference wrapper around :class:`SocialLSTM`."""

    def __init__(
        self,
        model: SocialLSTM,
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
        """Predict future trajectories.

        Args:
            observed_trajectory: Observed positions ``(T_obs, 2)`` for the
                primary agent.
            context: May include ``"neighbor_trajectories"`` with shape
                ``(T_obs, N_neighbors, 2)``.
        """
        self.model.eval()
        context = context or {}

        obs = torch.tensor(observed_trajectory, dtype=torch.float32, device=self.device)
        obs = obs.unsqueeze(1)  # (T_obs, 1, 2)

        # Optionally include neighbours.
        if "neighbor_trajectories" in context:
            neighbors = torch.tensor(
                context["neighbor_trajectories"], dtype=torch.float32, device=self.device
            )
            obs = torch.cat([obs, neighbors], dim=1)  # (T_obs, 1+N_neigh, 2)

        # Sample by running forward multiple times (Gaussian sampling).
        all_trajs: list[np.ndarray] = []
        for _ in range(self.num_samples):
            means, params = self.model(obs, pred_horizon=self.horizon)
            # Sample from the predicted Gaussian.
            mu_x = params[:, 0, 0]
            mu_y = params[:, 0, 1]
            sigma_x = params[:, 0, 2]
            sigma_y = params[:, 0, 3]
            rho = params[:, 0, 4]

            sampled = torch.zeros(self.horizon, 2, device=self.device)
            for t in range(self.horizon):
                mean = torch.tensor([mu_x[t], mu_y[t]], device=self.device)
                cov = torch.tensor(
                    [
                        [sigma_x[t] ** 2, rho[t] * sigma_x[t] * sigma_y[t]],
                        [rho[t] * sigma_x[t] * sigma_y[t], sigma_y[t] ** 2],
                    ],
                    device=self.device,
                )
                dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
                sampled[t] = dist.sample()
            all_trajs.append(sampled.cpu().numpy())

        trajectories = np.stack(all_trajs, axis=0)  # (N_samples, T, 2)
        probabilities = np.ones(self.num_samples) / self.num_samples
        timestamps = np.arange(1, self.horizon + 1) * self.dt

        return PredictionResult(
            trajectories=trajectories,
            probabilities=probabilities,
            timestamps=timestamps,
        )
