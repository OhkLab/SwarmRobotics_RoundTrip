# ライブラリのインポート
import numpy as np
import torch
import argparse

from solutions.cnn import CNN
from mlagents_envs.environment import ActionTuple, UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app-path', help='Path to unity application', default='Applications/20robots_default')
    parser.add_argument('--load-model', help='Path to model', default='pretrained/model.npz')
    parser.add_argument('--timesteps', help='Timesteps per episode', type=int, default=3000)
    parser.add_argument('--reps', help='Repeats number', type=int, default=1)
    config, _ = parser.parse_known_args()
    return config


def main(config):
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(width=400, height=400, time_scale=10, capture_frame_rate=50)
    channel2 = EnvironmentParametersChannel()
    channel2.set_float_parameter("time_steps", float(config.timesteps))
    env = UnityEnvironment(file_name=config.app_path, side_channels=[channel1, channel2])

    agent = CNN(
        device=torch.device('cpu'),
        file_name=config.app_path,
        act_dim=4,
        feat_dim=128
    )
    agent.load(config.load_model)

    env.reset()
    behavior_names = list(env.behavior_specs.keys())
    decision_steps, terminal_steps = env.get_steps(behavior_names[0])
    while True:
        for i in decision_steps.agent_id:
            obs = decision_steps.obs[0][i] * 255
            action = agent.get_action(obs)
            if i == 0:
                agent.show_gui(obs, target=0)
            action_tuple = ActionTuple(continuous=np.expand_dims(action, axis=0))
            env.set_action_for_agent(behavior_names[0], i, action_tuple)
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_names[0])
        if len(terminal_steps.interrupted) > 0:
            env.reset()
            agent.reset()


if __name__ == '__main__':
    args = parse_args()
    main(args)