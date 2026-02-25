import os

import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

# Register robotics envs
gym.register_envs(gymnasium_robotics)


def make_train_env():
    env = gym.make("FetchReach-v4", max_episode_steps=50)
    env = Monitor(env)
    return env


def make_eval_env(video_folder="./videos_eval", record_every_eval_episodes=1):
    """
    record_every_eval_episodes=1  -> record setiap episode evaluasi
    record_every_eval_episodes=5  -> record episode eval ke 0,5,10,...
    """
    os.makedirs(video_folder, exist_ok=True)

    env = gym.make("FetchReach-v4", max_episode_steps=50, render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="fetchreach_eval",
        episode_trigger=lambda ep: (ep % record_every_eval_episodes == 0),
    )

    env = Monitor(env)
    return env


def main():
    train_env = make_train_env()
    eval_env = make_eval_env(video_folder="./videos_eval", record_every_eval_episodes=1)

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.95,
        tensorboard_log="./fetchreach_tensorboard/",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/results/",
        eval_freq=10_000,  # evaluasi tiap 10k timesteps
        n_eval_episodes=5,  # jumlah episode evaluasi
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=200_000, callback=eval_callback)

    model.save("fetch_reach_her_sac")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
