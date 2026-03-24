from collections import deque
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import scipy.ndimage
import torch as th
import torch.nn.functional as F

from ..utils import Rotation3


class OnlineLocalEsdfBuilder:
    def __init__(
        self,
        num_envs: int,
        sensor_cfg: Dict,
        device: th.device,
        kwargs: Optional[Dict] = None,
    ):
        kwargs = kwargs or {}
        self.num_envs = num_envs
        self.device = th.device(device)
        self.sensor_uuid = sensor_cfg["uuid"]
        self.window = int(kwargs.get("window", 4))
        self.update_interval = int(kwargs.get("update_interval", 2))
        self.grid_resolution = float(kwargs.get("grid_resolution", 0.25))
        self.grid_size = tuple(int(v) for v in kwargs.get("grid_size", [32, 32, 16]))
        self.projection_size = tuple(
            int(v) for v in kwargs.get("projection_size", [12, 16])
        )
        self.clip_distance = float(kwargs.get("clip_distance", 2.0))
        self.sensor_near = float(sensor_cfg.get("near", 0.25))
        self.sensor_far = float(sensor_cfg.get("far", 20.0))
        self.sensor_hfov_deg = float(sensor_cfg.get("hfov", 89.0))

        bounds = kwargs.get("grid_bounds", None)
        if bounds is None:
            self.min_bounds = th.tensor([0.0, -4.0, -2.0], dtype=th.float32)
            self.max_bounds = th.tensor([8.0, 4.0, 2.0], dtype=th.float32)
        else:
            self.min_bounds = th.tensor(bounds["min"], dtype=th.float32)
            self.max_bounds = th.tensor(bounds["max"], dtype=th.float32)

        sensor_position = th.tensor(
            sensor_cfg.get("position", [0.0, 0.0, 0.0]), dtype=th.float32
        )
        sensor_orientation = th.tensor(
            sensor_cfg.get("orientation", [0.0, 0.0, 0.0]), dtype=th.float32
        ).unsqueeze(0)
        self.sensor_pos_body = sensor_position
        self.sensor_rot_body = Rotation3.from_euler_zyx(sensor_orientation).R[0]

        resolution = sensor_cfg.get("resolution", [72, 128])
        self.depth_height = int(resolution[0])
        self.depth_width = int(resolution[1])
        self._sensor_rays = self._build_sensor_rays()

        self._frames = [deque(maxlen=self.window) for _ in range(self.num_envs)]
        self._frame_counts = np.zeros((self.num_envs,), dtype=np.int32)
        self._local_esdf = th.full(
            (self.num_envs, *self.grid_size),
            self.clip_distance,
            dtype=th.float32,
            device=self.device,
        )
        self._esdf_proj = th.ones(
            (self.num_envs, 1, *self.projection_size),
            dtype=th.float32,
            device=self.device,
        )

    def _build_sensor_rays(self) -> th.Tensor:
        height = self.depth_height
        width = self.depth_width
        hfov = np.deg2rad(self.sensor_hfov_deg)
        vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * float(height) / float(width))
        fx = (width / 2.0) / np.tan(hfov / 2.0)
        fy = (height / 2.0) / np.tan(vfov / 2.0)

        xs = th.arange(width, dtype=th.float32) - ((width - 1) / 2.0)
        ys = th.arange(height, dtype=th.float32) - ((height - 1) / 2.0)
        v, u = th.meshgrid(ys, xs, indexing="ij")

        rays = th.stack(
            [
                th.ones_like(u),
                -(u / fx),
                -(v / fy),
            ],
            dim=-1,
        )
        return rays

    def reset(self, indices: Optional[Iterable[int]] = None):
        if indices is None:
            indices = range(self.num_envs)
        for idx in indices:
            self._frames[int(idx)].clear()
            self._frame_counts[int(idx)] = 0
        index_tensor = th.as_tensor(list(indices), dtype=th.long, device=self.device)
        if index_tensor.numel() > 0:
            self._local_esdf[index_tensor] = self.clip_distance
            self._esdf_proj[index_tensor] = 1.0

    def update(
        self,
        depth_batch: th.Tensor,
        positions: th.Tensor,
        rotations_wb: th.Tensor,
        indices: Optional[Sequence[int]] = None,
    ):
        if indices is None:
            indices = list(range(self.num_envs))
        else:
            indices = [int(v) for v in indices]

        for batch_idx, env_idx in enumerate(indices):
            frame = {
                "depth": depth_batch[batch_idx, 0].detach().to("cpu", dtype=th.float32),
                "position": positions[batch_idx].detach().to("cpu", dtype=th.float32),
                "rotation_wb": rotations_wb[batch_idx]
                .detach()
                .to("cpu", dtype=th.float32),
            }
            self._frames[env_idx].append(frame)
            self._frame_counts[env_idx] += 1
            if (
                self._frame_counts[env_idx] == 1
                or self._frame_counts[env_idx] % self.update_interval == 0
            ):
                self._rebuild_env(env_idx, frame["position"], frame["rotation_wb"])

    def _rebuild_env(
        self, env_idx: int, current_position: th.Tensor, current_rotation_wb: th.Tensor
    ):
        occupancy = np.zeros(self.grid_size, dtype=bool)
        current_rotation_bw = current_rotation_wb.transpose(0, 1)
        min_bounds = self.min_bounds.numpy()
        max_bounds = self.max_bounds.numpy()

        for frame in self._frames[env_idx]:
            points_body_hist = self._depth_to_body_points(frame["depth"])
            if points_body_hist.numel() == 0:
                continue

            world_points = (
                frame["rotation_wb"] @ points_body_hist.transpose(0, 1)
            ).transpose(0, 1) + frame["position"]
            body_points = (
                current_rotation_bw @ (world_points - current_position).transpose(0, 1)
            ).transpose(0, 1)

            body_points_np = body_points.numpy()
            in_bounds = np.all(
                (body_points_np >= min_bounds) & (body_points_np <= max_bounds), axis=1
            )
            if not np.any(in_bounds):
                continue

            body_points_np = body_points_np[in_bounds]
            grid_coords = np.floor(
                (body_points_np - min_bounds[None]) / self.grid_resolution
            ).astype(np.int32)
            grid_coords[:, 0] = np.clip(grid_coords[:, 0], 0, self.grid_size[0] - 1)
            grid_coords[:, 1] = np.clip(grid_coords[:, 1], 0, self.grid_size[1] - 1)
            grid_coords[:, 2] = np.clip(grid_coords[:, 2], 0, self.grid_size[2] - 1)
            occupancy[
                grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]
            ] = True

        esdf_np = scipy.ndimage.distance_transform_edt(~occupancy).astype(np.float32)
        esdf_np *= self.grid_resolution
        esdf_np = np.clip(esdf_np, 0.0, self.clip_distance)
        esdf = th.from_numpy(esdf_np).to(self.device)
        self._local_esdf[env_idx] = esdf
        self._esdf_proj[env_idx] = self._project_esdf(esdf)

    def _depth_to_body_points(self, depth: th.Tensor) -> th.Tensor:
        valid = (depth > self.sensor_near) & (depth < (self.sensor_far - 1e-3))
        if not th.any(valid):
            return th.empty((0, 3), dtype=th.float32)

        points_sensor = self._sensor_rays[valid] * depth[valid].unsqueeze(-1)
        points_body = (
            self.sensor_rot_body @ points_sensor.transpose(0, 1)
        ).transpose(0, 1) + self.sensor_pos_body
        return points_body

    def _project_esdf(self, esdf: th.Tensor) -> th.Tensor:
        bev = esdf.min(dim=2).values.unsqueeze(0).unsqueeze(0)
        bev = -F.adaptive_max_pool2d(-bev, self.projection_size)
        bev = bev.clamp(min=0.0, max=self.clip_distance)
        bev = (bev / self.clip_distance) * 2.0 - 1.0
        return bev.squeeze(0)

    def get_projection(self, indices: Optional[Sequence[int]] = None) -> th.Tensor:
        if indices is None:
            return self._esdf_proj
        return self._esdf_proj[indices]

    def query(
        self,
        points_body: th.Tensor,
        env_indices: Optional[Sequence[int]] = None,
    ) -> th.Tensor:
        if points_body.ndim == 2:
            points_body = points_body.unsqueeze(0)
        if env_indices is None:
            if points_body.shape[0] != self.num_envs:
                raise ValueError("points_body batch size must match num_envs")
            env_indices = list(range(self.num_envs))
        env_indices = th.as_tensor(env_indices, dtype=th.long, device=self.device)
        if points_body.shape[0] != len(env_indices):
            raise ValueError("points_body batch size must match env_indices length")

        points_body = points_body.to(self.device, dtype=th.float32)
        original_shape = points_body.shape[:-1]
        points_flat = points_body.reshape(points_body.shape[0], -1, 3)

        coords = (points_flat - self.min_bounds.to(self.device)) / self.grid_resolution
        grid_shape = th.tensor(self.grid_size, dtype=th.float32, device=self.device)
        norm = (coords / (grid_shape - 1.0)) * 2.0 - 1.0
        grid = norm[:, None, None, :, [1, 0, 2]]

        volume = self._local_esdf[env_indices].permute(0, 3, 1, 2).unsqueeze(1)
        queried = F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        queried = queried.squeeze(1).squeeze(1).squeeze(1)
        return queried.reshape(*original_shape)

