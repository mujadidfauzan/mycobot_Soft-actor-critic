import argparse
import os
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from torch import nn

from source.envs import GraspingEnv, ReachingEnv, GraspingEnvV1

parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["GraspingEnv", "ReachingEnv"], default="GraspingEnv")
parser.add_argument("--timesteps", type=int, default=1_000_000)
parser.add_argument("--debug-view", action="store_true")
parser.add_argument(
    "--grasp-dataset",
    default=None,
    help="Path to a .npz file with 'qpos' and 'qvel' arrays (used by ReachingEnv).",
)
args = parser.parse_args()

run_name = f"SAC_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
env_name = args.env
DEBUG_VIEW = bool(args.debug_view)
models_dir = os.path.join("logs", "models", env_name, run_name)
videos_dir = os.path.join("logs", "videos", env_name, run_name)
tb_dir = os.path.join("logs", "tensorboard", env_name, run_name)

os.makedirs(models_dir, exist_ok=True)
os.makedirs(videos_dir, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "source", "robot", "object_lift.xml")
run_prefix = "grasp" if env_name == "GraspingEnv" else "reach"


def make_env():
    if env_name == "ReachingEnv":
        env = ReachingEnv(
            xml_file=model_path,
            render_mode="human" if DEBUG_VIEW else "rgb_array",
            grasp_state_dataset_path=args.grasp_dataset,
        )
    elif env_name == "GraspingEnv":
        env = GraspingEnvV1(
            xml_file=model_path,
            render_mode="human" if DEBUG_VIEW else "rgb_array",
        )

    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
video_folder = videos_dir

if not DEBUG_VIEW:
    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step % 50000 == 0,
        video_length=1000,
        name_prefix=run_prefix,
    )


checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path=models_dir,
    name_prefix=f"sac_{run_prefix}",
    save_replay_buffer=False,
    save_vecnormalize=False,
)


class RenderCallback(BaseCallback):
    def _on_training_start(self) -> None:
        env0 = self.training_env.envs[0].unwrapped
        env0.render()
        env0.enable_frame_visualization()

    def _on_step(self) -> bool:
        env0 = self.training_env.envs[0].unwrapped
        env0.render()
        return True


class InfoTensorboardCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._excluded_keys = {"object_key"}

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if len(infos) > 0:
                info0 = infos[0]
                for key, value in info0.items():
                    if key in self._excluded_keys:
                        continue
                    if isinstance(value, bool):
                        self.logger.record(f"custom/{key}", float(value))
                    elif isinstance(value, (int, float)):
                        self.logger.record(f"custom/{key}", float(value))
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
    total_timesteps=int(args.timesteps),
    callback=[
        checkpoint_callback,
        InfoTensorboardCallback(log_freq=1000),
    ],
)


model.save(os.path.join(models_dir, f"sac_{run_prefix}_final"))

env.close()
