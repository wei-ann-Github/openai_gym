"""
Documentation: https://pettingzoo.farama.org/environments/classic/connect_four/
Petting Zoo API: https://github.com/Farama-Foundation/PettingZoo#api
"""
import matplotlib.pyplot as plt

import gymnasium as gym
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env()
env.reset()
env.render()
# agents= ['player_0', 'player_1']

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
    print(action)
    env.step(action)

def _show_state(env, step=0, info=""):
    # env.reset()

    plt.figure(3)
    plt.clf()
    plt.imshow(env.render())
    plt.title("%s | Step: %d %s" % (env._spec.id, step, info))
    plt.axis('off')

    plt.show()

# env.reset()
# _show_state(env, step=5)
# env.render()
env.close()
