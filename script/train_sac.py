import os
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from torch import nn

from source.envs import GraspingEnv

run_name = f"SAC_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"

models_dir = os.path.join("logs", "models", run_name)
videos_dir = os.path.join("logs", "videos", run_name)
tb_dir = os.path.join("logs", "tensorboard", run_name)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True)


def make_env():
    env = GraspingEnv(
        xml_file="source/robot/object_lift.xml",
        render_mode="rgb_array",
    )

    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])

video_folder = videos_dir


env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda step: step % 50000 == 0,
    video_length=1000,
    name_prefix="lift_reach",
)


checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path=models_dir,
    name_prefix="sac_lift",
    save_replay_buffer=False,
    save_vecnormalize=False,
)


class InfoTensorboardCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if len(infos) > 0:
                info0 = infos[0]
                for k in [
                    "dist",
                    "reward_dist",
                    # "touch_bonus",
                    "control_penalty",
                    "reward_target",
                    # "reward_lift",
                    # "reward_orient",
                    "reward_dist_tanh",
                    "reward_target_tanh",
                ]:
                    if k in info0:
                        self.logger.record(f"custom/{k}", float(info0[k]))
        return True


policy_kwargs = dict(
    net_arch=dict(
        pi=[512, 512, 256],
        qf=[512, 512, 256],
    ),
    activation_fn=nn.ReLU,
)

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    learning_starts=5000,
    batch_size=256,
    ent_coef="auto_0.01",
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    tensorboard_log=tb_dir,
    device="auto",
    policy_kwargs=policy_kwargs,
)


model.learn(
    total_timesteps=1_000_000,
    callback=[checkpoint_callback, InfoTensorboardCallback(log_freq=1000)],
)


model.save(os.path.join(models_dir, "sac_lift_final"))

env.close()
