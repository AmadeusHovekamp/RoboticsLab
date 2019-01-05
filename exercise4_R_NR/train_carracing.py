# export DISPLAY=:0

import sys
sys.path.append("../")

import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import EpisodeStats
from carracing_utils import (rgb2gray,
                             STRAIGHT,
                             LEFT,
                             RIGHT,
                             ACCELERATE,
                             BRAKE,
                             one_hot,
                             action_to_id,
                             id_to_action,
                             display_state)


import argparse

def run_episode(env, agent, current_episode, deterministic, skip_frames=0,
                do_training=True, testing=False, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    current_frame = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            current_frame += 1
            reward += r

            if not testing and args.verbose and args.show_frame_reward != 0 and ((current_frame % args.show_frame_reward) == 0):
                print("step reward in frame {:4d}: {:4.1f}".format(current_frame, reward))

            if rendering or (not testing and (args.render_frequency != 0) and (current_episode % args.render_frequency == 0)):
                env.render()

            if terminal:
                 break

        next_state = state_preprocessing(next_state)

        if current_frame % 10 == 0:
            display_state(next_state)

        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)
            pass

        stats.step(reward, action_id)

        if not testing and args.verbose and args.show_frame_reward != 0 and ((current_frame % args.show_frame_reward) == 0):
            print("stats in frame {:4d}:\t{}".format(current_frame, stats))

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps :
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0,
                 model_dir=os.path.join(".", "models", "CarRacing-v0"),
                 tensorboard_dir=os.path.join(".", "tensorboard", "CarRacing-v0")):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"),
                             ["episode_reward", "straight", "left", "right", "accel", "brake", "eval_reward"])

    for i in range(num_episodes):

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        # stats = run_episode(env, agent, i, skip_frames=5, deterministic=False, do_training=True, rendering=False)
        if args.verbose:
            print("\n\nEpisode {}".format(i))
        stats = run_episode(env, agent, i, skip_frames=3, deterministic=False,
                            do_training=True, rendering=False, max_timesteps=300,
                            history_length=history_length)

        if args.verbose:
            print("stats in eposide {:4d}:\t{}\t\tReplayBuffer{}".format(i, stats, len(agent.replay_buffer._data.states)))


        # evaluate agent with deterministic actions from time to time
        if i % 10 == 0 or i >= (num_episodes - 1):
            if args.verbose:
                print("\nStarting evaluation...")

            evaluation_stats = []
            for j in range(1):
                eval_stats = run_episode(env, agent, i, deterministic=False, do_training=False, rendering=True,
                                         history_length=history_length)
                evaluation_stats.append(eval_stats.episode_reward)
            mean_reward = np.mean(evaluation_stats)

            if args.verbose:
                print("evaluation: mean_reward after eposide {}:  \t{}".format(i, mean_reward))
                print("            ", eval_stats)

        if ((not i == 0) and (args.checkpoint_frequency != 0) and (i % args.checkpoint_frequency == 0)) or (i >= num_episodes - 1):
            save_agent(agent, model_dir)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                "straight" : stats.get_action_usage(STRAIGHT),
                "left" : stats.get_action_usage(LEFT),
                "right" : stats.get_action_usage(RIGHT),
                "accel" : stats.get_action_usage(ACCELERATE),
                "brake" : stats.get_action_usage(BRAKE),
                "eval_reward" : mean_reward
            })


    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

def save_agent(agent, model_dir, model_name="dqn_agent"):
    print("Saving agent...")
    agent.saver.save(agent.sess, os.path.join(model_dir, model_name + ".ckpt"))
    print("... Agent saved!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", default=False, help="Print episode statistics.")
    parser.add_argument("--show_frame_reward", default=0, type=int, nargs=1, help="Displays current reward every n frames (default = 10).")
    parser.add_argument("--render_frequency", default=5, type=int, nargs=1, help="The frequency of rendering the graphical output (default = 5).")
    parser.add_argument("--checkpoint_frequency", default=20, type=int, nargs=1, help="The frequency of creating and saving model checkpoints (default = 20).")
    parser.add_argument("--path_for_reloading", default=os.path.join(".", "models", "CarRacing-v0", "dqn_agent.ckpt"), type=str, nargs="?",
                        help="Path where the agent is located.")
    parser.add_argument("--fullrun", action="store_true", default=False, help="Run until episode limit.")
    args = parser.parse_args()

    game_name = "CarRacing-v0"
    print("Starting {} ...  ".format(game_name))

    env = gym.make(game_name).unwrapped

    # Define Q network, target network and DQN agent
    num_states = 96
    num_actions = 5
    num_episodes = 500
    history_length = 2

    q_net = CNN(num_states, num_actions, lr = 0.001, history_length=history_length)
    target_net = CNNTargetNetwork(num_states, num_actions, lr = 0.001, history_length=history_length)

    # init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(game_name, q_net, target_net, num_actions, batch_size=64, epsilon=0.15)

    # Load stored model
    if args.path_for_reloading != "":
        agent.load(args.path_for_reloading)

    # train DQN agent with train_online(...)
    train_online(env, agent, num_episodes=100, history_length=history_length)
