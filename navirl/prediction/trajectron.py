from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as _err:
    raise ImportError(
        "PyTorch is required for Trajectron. Install it with: pip install torch"
    ) from _err

from navirl.prediction.base import PredictionResult, TrajectoryPredictor


class _NodeEncoder(nn.Module):
    """Encodes the history of a single node (agent)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Encode agent history.

        Args:
            history: ``(batch, T, input_dim)``

        Returns:
            ``(batch, hidden_dim)``
        """
        _, (h, _) = self.lstm(history)
        return h[-1]  # (batch, hidden_dim)


class _EdgeEncoder(nn.Module):
    """Encodes interaction features along an edge between two nodes."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Encode edge features.

        Args:
            edge_features: ``(batch, input_dim)`` — typically relative position
                and velocity.

        Returns:
            ``(batch, hidden_dim)``
        """
        return self.mlp(edge_features)


class _CVAEDecoder(nn.Module):
    """CVAE decoder that produces a GMM over future positions.

    The latent variable *z* combined with the conditioning context is
    decoded into Gaussian mixture parameters at each future time-step.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        context_dim: int = 192,
        hidden_dim: int = 128,
        output_dim: int = 2,
        pred_horizon: int = 12,
        num_components: int = 5,
    ) -> None:
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_components = num_components
        self.output_dim = output_dim

        self.lstm = nn.LSTMCell(latent_dim + context_dim, hidden_dim)
        # Per-component: mean (2) + log_sigma (2) + weight logit (1) => 5 per component
        gmm_param_dim = num_components * (output_dim + output_dim + 1)
        self.gmm_head = nn.Linear(hidden_dim, gmm_param_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self, z: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent + context into GMM parameters at each step.

        Returns:
            means: ``(batch, T, K, 2)``
            log_sigmas: ``(batch, T, K, 2)``
            log_weights: ``(batch, T, K)``
        """
        batch = z.size(0)
        h = torch.zeros(batch, self.hidden_dim, device=z.device)
        c = torch.zeros(batch, self.hidden_dim, device=z.device)
        inp = torch.cat([z, context], dim=1)

        means_list, lsig_list, lw_list = [], [], []
        for _ in range(self.pred_horizon):
            h, c = self.lstm(inp, (h, c))
            raw = self.gmm_head(h)  # (batch, K*5)
            raw = raw.view(batch, self.num_components, self.output_dim * 2 + 1)
            mu = raw[:, :, : self.output_dim]
            log_sigma = raw[:, :, self.output_dim : 2 * self.output_dim]
            log_w = raw[:, :, -1]

            means_list.append(mu)
            lsig_list.append(log_sigma)
            lw_list.append(log_w)

        means = torch.stack(means_list, dim=1)
        log_sigmas = torch.stack(lsig_list, dim=1)
        log_weights = torch.stack(lw_list, dim=1)
        return means, log_sigmas, log_weights


class Trajectron(nn.Module):
    """Simplified Trajectron++-style model.

    Combines:
    - Per-node history encoding (LSTM)
    - Edge interaction encoding (MLP)
    - CVAE for multimodal future prediction
    - GMM output distribution
    """

    def __init__(
        self,
        input_dim: int = 2,
        node_hidden_dim: int = 128,
        edge_hidden_dim: int = 64,
        latent_dim: int = 32,
        pred_horizon: int = 12,
        num_gmm_components: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.pred_horizon = pred_horizon

        self.node_encoder = _NodeEncoder(input_dim, node_hidden_dim)
        self.edge_encoder = _EdgeEncoder(input_dim * 2, edge_hidden_dim)

        context_dim = node_hidden_dim + edge_hidden_dim

        # CVAE: recognition network q(z|x, y) and prior p(z|x)
        self.recognition_net = nn.Sequential(
            nn.Linear(context_dim + input_dim * pred_horizon, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),  # mean + log_var
        )
        self.prior_net = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )
        self.decoder = _CVAEDecoder(
            latent_dim, context_dim, node_hidden_dim, input_dim, pred_horizon, num_gmm_components
        )

    def _encode_context(
        self,
        node_history: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode node history and edge interactions into a context vector.

        Args:
            node_history: ``(batch, T_obs, 2)``
            edge_features: ``(batch, edge_input_dim)`` or ``None``

        Returns:
            ``(batch, context_dim)``
        """
        node_enc = self.node_encoder(node_history)  # (batch, node_hidden)
        if edge_features is not None:
            edge_enc = self.edge_encoder(edge_features)
        else:
            edge_enc = torch.zeros(node_enc.size(0), 64, device=node_enc.device)
        return torch.cat([node_enc, edge_enc], dim=1)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        node_history: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        During training, *future* should be provided so that the
        recognition network can be used for the ELBO.

        Args:
            node_history: ``(batch, T_obs, 2)``
            future: ``(batch, T_pred, 2)`` — ground-truth future (training only).
            edge_features: ``(batch, edge_dim)`` — optional edge features.

        Returns:
            Dictionary with ``means``, ``log_sigmas``, ``log_weights`` and,
            during training, ``kl_loss``.
        """
        context = self._encode_context(node_history, edge_features)
        batch = context.size(0)

        prior_params = self.prior_net(context)
        prior_mu, prior_log_var = prior_params.chunk(2, dim=1)

        if future is not None:
            # Recognition network.
            future_flat = future.reshape(batch, -1)
            recog_input = torch.cat([context, future_flat], dim=1)
            recog_params = self.recognition_net(recog_input)
            recog_mu, recog_log_var = recog_params.chunk(2, dim=1)
            z = self._reparameterize(recog_mu, recog_log_var)

            # KL divergence.
            kl = -0.5 * torch.sum(
                1 + recog_log_var - prior_log_var
                - (recog_log_var.exp() + (recog_mu - prior_mu) ** 2) / prior_log_var.exp(),
                dim=1,
            ).mean()
        else:
            z = self._reparameterize(prior_mu, prior_log_var)
            kl = torch.tensor(0.0, device=context.device)

        means, log_sigmas, log_weights = self.decoder(z, context)

        result: Dict[str, torch.Tensor] = {
            "means": means,
            "log_sigmas": log_sigmas,
            "log_weights": log_weights,
        }
        if future is not None:
            result["kl_loss"] = kl
        return result

    @staticmethod
    def gmm_nll(
        means: torch.Tensor,
        log_sigmas: torch.Tensor,
        log_weights: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood under the predicted GMM.

        Args:
            means: ``(batch, T, K, 2)``
            log_sigmas: ``(batch, T, K, 2)``
            log_weights: ``(batch, T, K)``
            target: ``(batch, T, 2)``

        Returns:
            Scalar loss.
        """
        target = target.unsqueeze(2)  # (batch, T, 1, 2)
        sigmas = torch.exp(log_sigmas)
        diff = target - means  # (batch, T, K, 2)
        log_p = (
            -0.5 * ((diff / (sigmas + 1e-6)) ** 2).sum(dim=-1)
            - log_sigmas.sum(dim=-1)
            - np.log(2 * np.pi)
        )
        log_w = F.log_softmax(log_weights, dim=-1)
        log_p_mixture = torch.logsumexp(log_p + log_w, dim=-1)  # (batch, T)
        return -log_p_mixture.mean()


class TrajectronPredictor(TrajectoryPredictor):
    """Inference wrapper around :class:`Trajectron`."""

    def __init__(
        self,
        model: Trajectron,
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
        context: Optional[Dict[str, Any]] = None,
    ) -> PredictionResult:
        self.model.eval()
        context = context or {}

        obs = torch.tensor(observed_trajectory, dtype=torch.float32, device=self.device)
        obs = obs.unsqueeze(0)  # (1, T_obs, 2)

        edge_features = None
        if "edge_features" in context:
            edge_features = torch.tensor(
                context["edge_features"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        all_trajs: List[np.ndarray] = []
        for _ in range(self.num_samples):
            out = self.model(obs, future=None, edge_features=edge_features)
            means = out["means"]  # (1, T, K, 2)
            log_sigmas = out["log_sigmas"]
            log_weights = out["log_weights"]  # (1, T, K)

            # Sample a component at each time-step.
            weights = F.softmax(log_weights[0], dim=-1)  # (T, K)
            traj = torch.zeros(self.horizon, 2, device=self.device)
            for t in range(self.horizon):
                k = torch.multinomial(weights[t], 1).item()
                mu = means[0, t, k]
                sigma = torch.exp(log_sigmas[0, t, k])
                traj[t] = mu + sigma * torch.randn(2, device=self.device)
            all_trajs.append(traj.cpu().numpy())

        trajectories = np.stack(all_trajs, axis=0)
        probabilities = np.ones(self.num_samples) / self.num_samples
        timestamps = np.arange(1, self.horizon + 1) * self.dt

        return PredictionResult(
            trajectories=trajectories,
            probabilities=probabilities,
            timestamps=timestamps,
        )
