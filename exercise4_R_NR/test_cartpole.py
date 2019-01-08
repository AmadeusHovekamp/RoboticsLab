import argparse
import os
from datetime import datetime
import gym
import json
from dqn.dqn_agent import DQNAgent
from train_cartpole import run_episode
from dqn.networks import *
import numpy as np
from tensorboard_evaluation import *

np.random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_path", default=os.path.join(".", "models", "CartPole-v0", "full_run.ckpt"), type=str, nargs="?",
                        help="Path where the agent is located.")
    args = parser.parse_args()


    tensorboard_dir=os.path.join(".", "tensorboard","CartPole-v0")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "test"),
                             ["episode_reward",
                              "a_0",
                              "a_1"])

    env = gym.make("CartPole-v0").unwrapped

    num_states = 4
    num_actions = env.action_space.n

    # load DQN agent

    q_net = NeuralNetwork(num_states, num_actions, lr = 0.001)
    target_net = TargetNetwork(num_states, num_actions, lr = 0.001)
    agent = DQNAgent("CarRacing-v0", q_net, target_net, num_actions)
    agent.load(args.agent_path)

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)
        print(i, "stats.episode_reward:\t" ,stats.episode_reward)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "a_0" : stats.get_action_usage(0),
                                                      "a_1" : stats.get_action_usage(1)
                                                     })

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    if not os.path.exists("./results"):
        os.mkdir("./results")

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    # close tensorboard session
    tensorboard.close_session()

    env.close()
    print('... finished')
