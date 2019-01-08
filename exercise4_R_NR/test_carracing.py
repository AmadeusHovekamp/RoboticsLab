from __future__ import print_function

import gym
from dqn.dqn_agent import DQNAgent
from train_carracing import run_episode
from dqn.networks import *
import numpy as np
import argparse
import os
from datetime import datetime
from tensorboard_evaluation import *
from carracing_utils import *
import json

np.random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default=os.path.join(".", "models", "CarRacing-v0", "dqn_agent.ckpt"), type=str, nargs="?",
                        help="Path where the agent is located.")
    args = parser.parse_args()

    print("Starting test_carracing...")

    tensorboard_dir=os.path.join(".", "tensorboard","CarRacing-v0")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "test"),
                             ["episode_reward", "straight", "left", "right", "accel", "brake"])


    env = gym.make("CarRacing-v0").unwrapped

    history_length =  1

    # Define networks and load agent
    # ....
    num_states = 96
    num_actions = 5

    game_name = "CarRacing_v0"
    q_net = CNN(num_states, num_actions, history_length=history_length)
    target_net = CNNTargetNetwork(num_states, num_actions, history_length=history_length)

    # init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(game_name, q_net, target_net, num_actions)

    # Load stored model
    agent.load(args.agent_path)


    n_test_episodes = 10

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, i, deterministic=True, do_training=False, testing=True, rendering=True, history_length=history_length)
        episode_rewards.append(stats.episode_reward)

        tensorboard.write_episode_data(i, eval_dict={
                "episode_reward" : stats.episode_reward,
                "straight" : stats.get_action_usage(STRAIGHT),
                "left" : stats.get_action_usage(LEFT),
                "right" : stats.get_action_usage(RIGHT),
                "accel" : stats.get_action_usage(ACCELERATE),
                "brake" : stats.get_action_usage(BRAKE),
            })

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()


    result_path = "./results/CarRacing-v0/"
    if not os.path.exists(result_path):
        os.mkdirs(result_path)

    fname = result_path + "results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
