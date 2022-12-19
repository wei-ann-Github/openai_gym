"""This is a minimal example of using Tianshou with MARL to train agents.
Example from: https://github.com/Farama-Foundation/PettingZoo/blob/74f1f1314bf7cb7e5846c94d765d4e07470de03a/tutorials/Tianshou/2_training_agents.py

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import os
from typing import Optional, Tuple

import gym
import numpy as np
from pathlib import Path
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net

from pettingzoo.classic import connect_four_v3

version = 'v2'

trained_path = Path('../log/rps/dqn/policy_v1.pth')
train_path = Path(f'../log/rps/dqn/policy_{version}.pth')

def _get_agents(
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        net = Net(
            state_shape=observation_space["observation"].shape
            or observation_space["observation"].n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )
        device = 'cuda'
        if not torch.cuda.is_available():
            device = 'cpu'
            
        agent_learn.load_state_dict(torch.load(trained_path, 
                                    map_location=torch.device(device))) 

    if agent_opponent is None:
        net = Net(
            state_shape=observation_space["observation"].shape
            or observation_space["observation"].n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_opponent = DQNPolicy(
            model=net,
            optim=optim,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

        device = 'cuda'
        if not torch.cuda.is_available():
            device = 'cpu'
            
        agent_opponent.load_state_dict(torch.load(trained_path, 
                                       map_location=torch.device(device))) 

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)

    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(connect_four_v3.env())


if __name__ == "__main__":

    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(10)])
    test_envs = DummyVectorEnv([_get_env for _ in range(10)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()
    print(f'policy: {policy}')

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        # model_save_path = os.path.join("log", "rps", "dqn", f"policy_{version}.pth")
        model_save_path = train_path
        os.makedirs(train_path.parent, exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.8

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50*10,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        # stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.01,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")
