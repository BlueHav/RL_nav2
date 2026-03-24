import torch as th
import numpy as np
from torch import nn
from typing import Type, Optional, Dict, Any, Union, List
from gymnasium import spaces

from .extractors import (
    FeatureExtractor,
    StateExtractor,
    StateTargetExtractor,
    ImageExtractor,
    StateImageExtractor,
    StateTargetImageExtractor,
    StateTargetDepthEsdfExtractor,
    create_mlp,
)
from .mlp_policy import MlpPolicy


# 多模态策略：特征提取器 + 可选时序模块 + MLP 动作头
class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_ih = nn.Linear(input_size, 3 * hidden_size)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size)
        self.ln_ih = nn.LayerNorm(3 * hidden_size)
        self.ln_hh = nn.LayerNorm(3 * hidden_size)

    def forward(self, x, h):
        gi = self.ln_ih(self.linear_ih(x))
        gh = self.ln_hh(self.linear_hh(h))
        i_r, i_z, i_n = gi.chunk(3, 1)
        h_r, h_z, h_n = gh.chunk(3, 1)

        r = th.sigmoid(i_r + h_r)
        z = th.sigmoid(i_z + h_z)
        n = th.tanh(i_n + r * h_n)
        h_next = (1 - z) * n + z * h
        return h_next


class MultiInputPolicy(MlpPolicy):
    """
    Builds an actor policy network with specifications from a dictionary.
    """

    feature_extractor_alias = {
        # "flatten": FlattenExtractor,
        "StateExtractor": StateExtractor,
        "ImageExtractor": ImageExtractor,
        "StateTargetExtractor": StateTargetExtractor,
        "StateImageExtractor": StateImageExtractor,
        "StateTargetImageExtractor": StateTargetImageExtractor,
        "StateTargetDepthEsdfExtractor": StateTargetDepthEsdfExtractor,
    }
    recurrent_alias = {
        "GRUCell": th.nn.GRUCell,
        "LayerNormGRUCell": LayerNormGRUCell,
        "LSTMCell": th.nn.LSTMCell,
    }

    def __init__(
        self,
        observation_space: spaces.Space,
        net_arch: Dict[str, List[int]],
        activation_fn: Union[str, nn.Module],
        output_activation_fn: Union[str, nn.Module],
        feature_extractor_class: Type[FeatureExtractor],
        output_activation_kwargs: Optional[Dict[str, Any]] = None,
        feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
        planner_head_kwargs: Optional[Dict[str, Any]] = None,
        device: th.device = "cuda",
    ):
        if isinstance(feature_extractor_class, str):
            feature_extractor_class = self.feature_extractor_alias[
                feature_extractor_class
            ]
        feature_extractor_kwargs = feature_extractor_kwargs or {}
        planner_head_kwargs = planner_head_kwargs or {}

        # get the size of features_dim before initializing MlpPolicy
        feature_extractor = feature_extractor_class(
            observation_space, **feature_extractor_kwargs
        )
        feature_norm = nn.LayerNorm(feature_extractor.features_dim)

        # add recurrent layer after feature_extractor
        _is_recurrent = False
        recurrent_state_type = "tensor"
        if net_arch.get("recurrent", None) is not None:
            _is_recurrent = True
            rnn_setting = net_arch.get("recurrent")
            rnn_class = rnn_setting.get("class")
            kwargs = rnn_setting.get("kwargs")

            if isinstance(rnn_class, str):
                rnn_class = self.recurrent_alias[rnn_class]

            recurrent_extractor = rnn_class(
                input_size=feature_extractor.features_dim, **kwargs
            )
            if rnn_class == th.nn.LSTMCell:
                recurrent_state_type = "lstm"
            in_dim = kwargs.get("hidden_size")
        else:
            in_dim = feature_extractor.features_dim

        planner_condition_dim = 4 if len(planner_head_kwargs) > 0 else 0

        super().__init__(
            in_dim + planner_condition_dim,
            net_arch,
            activation_fn,
            output_activation_fn,
            output_activation_kwargs,
            device,
        )

        self.feature_extractor = feature_extractor
        self.feature_norm = feature_norm
        if _is_recurrent:
            self._is_recurrent = True
            self._latent_dim = in_dim
            self.recurrent_extractor = recurrent_extractor
        self.recurrent_state_type = recurrent_state_type
        self.planner_head_kwargs = planner_head_kwargs
        self.planner_condition_dim = planner_condition_dim
        self._planner_head = None
        if len(self.planner_head_kwargs) > 0:
            self._build_planner_head(in_dim, activation_fn)

    def init_latent(self, batch_size: int, device: Optional[th.device] = None):
        device = device or self.device
        if not self.is_recurrent:
            return None
        if self.recurrent_state_type == "lstm":
            h = th.zeros((batch_size, self.latent_dim), device=device)
            c = th.zeros((batch_size, self.latent_dim), device=device)
            return h, c
        return th.zeros((batch_size, self.latent_dim), device=device)

    def mask_latent(self, latent, done_mask: th.Tensor):
        if not self.is_recurrent or latent is None:
            return latent

        latent_device = latent[0].device if self.recurrent_state_type == "lstm" else latent.device
        keep_mask = (~done_mask).to(dtype=th.float32, device=latent_device).unsqueeze(1)
        if self.recurrent_state_type == "lstm":
            h, c = latent
            return h * keep_mask, c * keep_mask
        return latent * keep_mask

    def detach_latent(self, latent):
        if latent is None:
            return None
        if self.recurrent_state_type == "lstm":
            h, c = latent
            return h.detach(), c.detach()
        return latent.detach()

    def _build_planner_head(self, input_dim: int, activation_fn):
        if isinstance(activation_fn, str):
            activation_fn = self.activation_fn_alias[activation_fn]

        yaw_bins = self.planner_head_kwargs.get("yaw_bins", [-60, -30, 0, 30, 60])
        pitch_bins = self.planner_head_kwargs.get("pitch_bins", [-20, 0, 20])
        num_candidates = int(
            self.planner_head_kwargs.get(
                "num_candidates", len(yaw_bins) * len(pitch_bins)
            )
        )
        if num_candidates != len(yaw_bins) * len(pitch_bins):
            raise ValueError("num_candidates must match yaw_bins x pitch_bins")

        planner_hidden = self.planner_head_kwargs.get("mlp_layer", [])
        self._planner_head = create_mlp(
            input_dim=input_dim,
            layer=planner_hidden,
            output_dim=num_candidates,
            activation_fn=activation_fn,
            batch_norm=self.planner_head_kwargs.get("bn", False),
            layer_norm=self.planner_head_kwargs.get("ln", False),
            device=self.device,
        )

        candidate_dirs = []
        for pitch_deg in pitch_bins:
            pitch = np.deg2rad(pitch_deg)
            for yaw_deg in yaw_bins:
                yaw = np.deg2rad(yaw_deg)
                candidate_dirs.append(
                    [
                        np.cos(pitch) * np.cos(yaw),
                        np.cos(pitch) * np.sin(yaw),
                        np.sin(pitch),
                    ]
                )
        self.register_buffer(
            "_planner_candidate_directions",
            th.as_tensor(candidate_dirs, dtype=th.float32),
        )

    def _extract_actor_features(self, obs, latent=None):
        features = self.feature_extractor(obs)
        features = self.feature_norm(features)
        if self.is_recurrent:
            if latent is None:
                latent = self.init_latent(features.shape[0], features.device)
            if self.recurrent_state_type == "lstm":
                h, c = latent
                h, c = self.recurrent_extractor(features, (h, c))
                latent = (h, c)
                return h, latent

            latent = self.recurrent_extractor(features, latent)
            return latent, latent

        return features, latent

    def _planner_condition(self, actor_features):
        aux_dict = {}
        conditioned_features = actor_features
        if self._planner_head is not None:
            planner_logits = self._planner_head(actor_features)
            planner_probs = th.softmax(planner_logits, dim=1)
            candidate_dirs = self.get_candidate_directions(actor_features.device)
            expected_dir = planner_probs @ candidate_dirs
            planner_conf = planner_probs.max(dim=1, keepdim=True).values
            conditioned_features = th.cat(
                [actor_features, expected_dir, planner_conf], dim=1
            )
            aux_dict = {
                "planner_logits": planner_logits,
                "planner_probs": planner_probs,
                "planner_expected_dir": expected_dir,
                "planner_confidence": planner_conf,
            }
        return conditioned_features, aux_dict

    def forward(self, obs, latent=None):
        actor_features, latent = self._extract_actor_features(obs, latent)
        conditioned_features, _ = self._planner_condition(actor_features)
        actions = super().forward(conditioned_features)
        if self.is_recurrent:
            return actions, latent
        return actions

    def forward_aux(self, obs, latent=None):
        actor_features, latent = self._extract_actor_features(obs, latent)
        conditioned_features, aux_dict = self._planner_condition(actor_features)
        actions = super().forward(conditioned_features)
        return actions, latent, aux_dict

    @property
    def has_planner_head(self):
        return self._planner_head is not None

    def get_candidate_directions(self, device: Optional[th.device] = None):
        candidate_directions = getattr(self, "_planner_candidate_directions", None)
        if candidate_directions is None:
            return None
        if device is None:
            return self._planner_candidate_directions
        return self._planner_candidate_directions.to(device)
