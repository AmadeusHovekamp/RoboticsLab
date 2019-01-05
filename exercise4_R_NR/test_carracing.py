from __future__ import print_function

import gym
from dqn.dqn_agent import DQNAgent
from train_carracing import run_episode
from dqn.networks import *
import numpy as np
import argparse
import os
from datetime import datetime

np.random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default=os.path.join(".", "models", "CarRacing-v0", "dqn_agent.ckpt"), type=str, nargs="?",
                        help="Path where the agent is located.")
    # parser.add_argument("--verbose", action="store_true", default=False, help="Print episode statistics.")
    # parser.add_argument("--show_frame_reward", default=10, type=int, nargs=1, help="Displays current reward every n frames (default = 10).")
    # parser.add_argument("--render_frequency", default=5, type=int, nargs=1, help="The frequency of rendering the graphical output (default = 5).")
    # parser.add_argument("--checkpoint_frequency", default=20, type=int, nargs=1, help="The frequency of creating and saving model checkpoints (default = 20).")
    # parser.add_argument("--path_for_reloading", default=os.path.join(".", "models", "CarRacing-v0", "dqn_agent.ckpt"), type=str, nargs="?",
    #                     help="Path where the agent is located.")
    # parser.add_argument("--fullrun", action="store_true", default=False, help="Run until episode limit.")
    args = parser.parse_args()

    print("Starting test_carracing...")

    env = gym.make("CarRacing-v0").unwrapped

    history_length =  0

    # Define networks and load agent
    # ....
    num_states = 96
    num_actions = 5
    num_episodes = 500

    game_name = "CarRacing_v0"
    q_net = CNN(num_states, num_actions, lr = 0.001)
    target_net = CNNTargetNetwork(num_states, num_actions, lr = 0.001)

    # init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(game_name, q_net, target_net, num_actions, batch_size=64, epsilon=0.1)

    # Load stored model
    agent.load(args.agent_path)


    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, i, deterministic=True, do_training=False, testing=True, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
