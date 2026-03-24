import os
import sys
import time
from typing import Any, Union, Optional, Type, Dict, TypeVar, ClassVar, List

import torch as th
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common import logger

from pylogtools import timerlog

timerlog.timer.print_logs(False)

from .debug import check_none_parameters, get_network_statistics, compute_gradient_norm
from .mlp_policy import MlpPolicy
from ..common import observation_to_device, rgba2rgb, ExitCode
from depthnav.scripts.eval_logger import Evaluate

#核心训练器：BPTT 算法，AdamW + 余弦 LR，梯度裁剪，TensorBoard 日志，checkpoint
class BPTT:
    """
    Back Propagation Through Time (BPTT).
    Pass in an env, eval_env, and policy
    Call learn() and get back the trained policy
    Gradients are computed across multiple timesteps and loss is computed at final time step.
    """

    def __init__(
        self,
        policy: Union[MlpPolicy],
        env: GymEnv,
        eval_envs: List[GymEnv] = None,
        eval_csvs: List[str] = None,
        learning_rate_init: float = 3e-4,
        learning_rate_final: float = 0.0,
        weight_decay: float = 0.0,
        run_name: Optional[str] = "BPTT",
        logging_dir: Optional[str] = "./saved",
        horizon: int = 32,
        gamma: float = 0.99,
        iterations: int = 1000,
        log_interval: int = 1,
        early_stop_reward_threshold: float = -th.inf,
        checkpoint_interval: int = 1000,
        device: Optional[Union[str, th.device]] = "cpu",
        teacher_score_kwargs: Optional[Dict[str, Any]] = None,
        planner_ce_weight: float = 0.3,
        planner_mse_weight: float = 0.1,
    ):
        self.logging_dir = os.path.abspath(logging_dir)
        self.run_path = self._create_run_path(run_name)

        self.env = env
        self.eval_envs = eval_envs or []
        self.eval_csvs = eval_csvs or []
        assert len(self.eval_envs) == len(self.eval_csvs)
        self.device = th.device(device)
        self.policy = policy.to(self.device)

        # training parameters
        self.iterations = iterations
        self.horizon = horizon
        self.gamma = gamma
        self.optimizer = th.optim.AdamW(
            self.policy.parameters(), lr=learning_rate_init, weight_decay=weight_decay
        )
        self.lr_schedule = th.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=iterations, eta_min=learning_rate_final
        )

        # training buffers used for logging
        self.log_interval = log_interval
        self.early_stop_reward_threshold = early_stop_reward_threshold
        self.checkpoint_interval = checkpoint_interval
        self.teacher_score_kwargs = teacher_score_kwargs or {}
        self.planner_ce_weight = planner_ce_weight
        self.planner_mse_weight = planner_mse_weight

        self.whitelisted_tensorboard_keys = [
            "avg_reward",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "avg_steps",
            "avg_path_length",
            "avg_speed",
            "max_speed",
            "max_acceleration",
            "max_yaw_rate",
            "avg_min_obstacle_distance",
        ]

        self.whitelisted_csv_keys = [
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "num_trials",
            "avg_speed",
            "max_speed",
            "avg_path_length",
            "avg_control_effort",
            "max_acceleration",
            "avg_yaw_rate",
            "max_yaw_rate",
            "avg_reward",
            "avg_min_obstacle_distance",
            "scene",
        ]

    def _create_run_path(self, run_name="BPTT"):
        index = 1
        run_name = run_name or "BPTT"
        while True:
            path = f"{self.logging_dir}/{run_name}_{index}"
            if not os.path.exists(path):
                break
            index += 1
        return path

    def save(self, filepath=None):
        filepath = filepath or self.run_path + ".pth"
        print(f"Saving to {filepath}")
        self.policy.save(filepath)

    def _normalize_candidate_scores(self, scores: th.Tensor) -> th.Tensor:
        score_min = scores.min(dim=1, keepdim=True).values
        score_max = scores.max(dim=1, keepdim=True).values
        return (scores - score_min) / (score_max - score_min + 1e-6)

    def learn(self, render=False, start_iter=0) -> ExitCode:
        """
        Train policy using BPTT, return True when finished training
        """
        assert self.horizon >= 1, "horizon must be greater than 1"
        assert self.env is not None

        self.policy.train()
        check_none_parameters(self.policy)
        self.start_time = time.time_ns()
        exit_code = ExitCode.ERROR
        self._logger = logger.configure(self.run_path, ["stdout", "tensorboard"])

        try:
            self.env.reset()
            episode_steps = 0
            latent_state = (
                self.policy.init_latent(self.env.num_envs, self.policy.device)
                if getattr(self.policy, "is_recurrent", False)
                and hasattr(self.policy, "init_latent")
                else None
            )
            for iter in tqdm(range(self.iterations)):
                timerlog.timer.tic("iteration")

                self.policy.train()
                reward_loss = th.zeros(
                    self.env.num_envs, dtype=th.float32, device=self.device
                )
                planner_ce_accum = th.tensor(0.0, device=self.device)
                planner_mse_accum = th.tensor(0.0, device=self.device)
                planner_steps = 0
                planner_metric_accum: Dict[str, float] = {}
                fps_start = time.time_ns()
                discount_factor = th.ones(
                    self.env.num_envs, dtype=th.float32, device=self.device
                )

                # reset agents once they have max_episode_steps experience
                if episode_steps >= self.env.max_episode_steps:
                    self.env.reset()
                    episode_steps = 0
                    latent_state = (
                        self.policy.init_latent(self.env.num_envs, self.policy.device)
                        if getattr(self.policy, "is_recurrent", False)
                        and hasattr(self.policy, "init_latent")
                        else None
                    )

                # rollout policy over horizon steps
                for _ in range(self.horizon):
                    obs = self.env.get_observation()
                    #坐标转换，habitat-sim坐标系转换为body坐标系
                    obs = observation_to_device(obs, self.policy.device)
                    #前向传播，计算动作
                    if type(self.policy) == MlpPolicy:
                        actions = self.policy(obs["state"])
                    else:
                        aux_dict = {}
                        next_latent_state = latent_state
                        if hasattr(self.policy, "forward_aux"):
                            if self.policy.is_recurrent:
                                actions, next_latent_state, aux_dict = self.policy.forward_aux(
                                    obs, latent_state
                                )
                            else:
                                actions, _, aux_dict = self.policy.forward_aux(obs)
                        else:
                            if self.policy.is_recurrent:
                                actions, next_latent_state = self.policy(obs, latent_state)
                            else:
                                actions = self.policy(obs)

                        if (
                            getattr(self.policy, "has_planner_head", False)
                            and "planner_logits" in aux_dict
                            and hasattr(self.env, "compute_planner_teacher")
                        ):
                            teacher = self.env.compute_planner_teacher(
                                self.policy.get_candidate_directions(self.env.device),
                                **self.teacher_score_kwargs,
                            )
                            if teacher is not None:
                                teacher_scores = teacher["scores"].to(self.device)
                                teacher_indices = teacher["best_indices"].to(self.device)
                                planner_logits = aux_dict["planner_logits"]
                                planner_ce = F.cross_entropy(
                                    planner_logits, teacher_indices
                                )
                                planner_mse = F.mse_loss(
                                    self._normalize_candidate_scores(planner_logits),
                                    teacher_scores,
                                )
                                planner_ce_accum = planner_ce_accum + planner_ce
                                planner_mse_accum = planner_mse_accum + planner_mse
                                planner_steps += 1
                                for metric_name, metric_value in teacher["metrics"].items():
                                    planner_metric_accum[metric_name] = (
                                        planner_metric_accum.get(metric_name, 0.0)
                                        + float(metric_value)
                                    )
                        latent_state = next_latent_state

                    # step
                    obs, reward, done, info = self.env.step(actions, is_test=False)
                    reward = reward.to(self.device)
                    done = done.to(self.device).to(th.bool)
                    reward_loss = reward_loss + -1.0 * reward * discount_factor

                    # if done, reset discount factor and latents
                    discount_factor = discount_factor * self.gamma * ~done + done
                    if getattr(self.policy, "is_recurrent", False):
                        if hasattr(self.policy, "mask_latent"):
                            latent_state = self.policy.mask_latent(latent_state, done)
                        elif latent_state is not None:
                            latent_state = latent_state * ~done.unsqueeze(1)

                episode_steps += self.horizon
                total_steps = self.env.num_envs * self.horizon
                time_elapsed = max(
                    (time.time_ns() - fps_start) / 1e9, sys.float_info.epsilon
                )
                fps = int(total_steps / time_elapsed)

                # backprop
                reward_loss = reward_loss / self.horizon  # average across the rollout
                reward_loss = reward_loss.mean()  # average across the batch
                planner_ce_term = (
                    planner_ce_accum / planner_steps
                    if planner_steps > 0
                    else th.tensor(0.0, device=self.device)
                )
                planner_mse_term = (
                    planner_mse_accum / planner_steps
                    if planner_steps > 0
                    else th.tensor(0.0, device=self.device)
                )
                loss = (
                    reward_loss
                    + self.planner_ce_weight * planner_ce_term
                    + self.planner_mse_weight * planner_mse_term
                )
                self.optimizer.zero_grad()
                loss.backward()

                # log total gradient magnitude and clip to prevent exploding gradients
                max_norm = 5.0
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_norm=max_norm
                )
                print(f"grad norm = {grad_norm:.4f}")

                # update policy
                self.optimizer.step()
                self.lr_schedule.step()

                # detach gradients
                self.env.detach()
                if hasattr(self.policy, "detach_latent"):
                    latent_state = self.policy.detach_latent(latent_state)
                elif latent_state is not None:
                    latent_state = latent_state.clone().detach()
                timerlog.timer.toc("iteration")

                timerlog.timer.tic("eval")
                if iter % self.log_interval == 0:
                    self.policy.eval()
                    for i, (eval_env, csv_file) in enumerate(
                        zip(self.eval_envs, self.eval_csvs)
                    ):
                        # evaluate policy in eval_env with multiple rollouts
                        e = Evaluate(eval_env, self.policy)
                        index = start_iter + iter
                        df = e.run_rollouts(
                            num_rollouts=5, run_name=index, render=render
                        )

                        # add a column to log the scene
                        df["scene"] = os.path.basename(
                            self.env.scene_manager.scene_path
                        )

                        # log the df to tensorboard
                        basename = os.path.basename(csv_file).split(".")[0]
                        self.df_to_tensorboard(self._logger, df, prefix=basename)

                        # save df to csv
                        write_header = not os.path.exists(csv_file)
                        df.to_csv(
                            csv_file,
                            float_format="%.3f",
                            mode="a",
                            header=write_header,
                            columns=self.whitelisted_csv_keys,
                        )
                        print(f"wrote stats to {csv_file}")

                        # check if we should early stop
                        last_avg_reward = df["avg_reward"].iloc[-1]
                        if last_avg_reward < self.early_stop_reward_threshold:
                            print("REWARD FELL BELOW EARLY STOP THRESHOLD")
                            print(f"Last reward: {last_avg_reward}")
                            exit_code = ExitCode.EARLY_STOP
                            for eval_env in self.eval_envs:
                                eval_env.close()
                            self.env.close()
                            return exit_code

                    # log and dump iter to tensorboard
                    self._logger.record(
                        "train/learning_rate", self.lr_schedule.get_last_lr()[0]
                    )
                    self._logger.record("train/loss", float(loss))
                    self._logger.record("train/reward_loss", float(reward_loss))
                    self._logger.record("train/planner_ce", float(planner_ce_term))
                    self._logger.record("train/planner_mse", float(planner_mse_term))
                    if planner_steps > 0:
                        for metric_name, metric_value in planner_metric_accum.items():
                            self._logger.record(
                                f"train/{metric_name}",
                                metric_value / float(planner_steps),
                            )
                    self._logger.record("train/grad_norm", float(grad_norm))
                    self._logger.record("train/steps_per_second", fps)
                    self._logger.dump(start_iter + iter)

                    timerlog.timer.toc("eval")
                    timerlog.timer.print_summary()
                    timerlog.timer.clear_history()
                if iter > 0 and iter % self.checkpoint_interval == 0:
                    self.save(self.run_path + "_iteration_" + str(iter) + ".pth")

            exit_code = ExitCode.SUCCESS
        except KeyboardInterrupt:
            self.save(self.run_path + "_iteration_" + str(iter) + ".pth")
            exit_code = ExitCode.KEYBOARD_INTERRUPT
        finally:
            for eval_env in self.eval_envs:
                eval_env.close()
            self.env.close()
        return exit_code

    def df_to_tensorboard(self, logger, df, prefix="eval"):
        keys = df.columns.tolist()
        for key in keys:
            if key not in self.whitelisted_tensorboard_keys:
                continue
            logger.record(f"{prefix}/{key}", df[key].iloc[-1])
