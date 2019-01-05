import argparse
import numpy as np
import gym
import itertools as it
from dqn.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from dqn.networks import NeuralNetwork, TargetNetwork
from utils import EpisodeStats
import time

def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            time.sleep(1/200)
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats

def train_online(env, agent, num_episodes, model_dir=os.path.join(".", "models","CartPole-v0"), tensorboard_dir=os.path.join(".", "tensorboard","CartPole-v0")):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("... train agent")

    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "a_0", "a_1", "eval_reward"])
    mean_reward = 0
    solved = False
    first_solved = True

    # training
    for i in range(num_episodes):
        stats = run_episode(env, agent, deterministic=False, do_training=True, rendering=False)

        if args.verbose:
            print("stats in eposide {:4d}:\t{}".format(i,stats))
        # evaluate your agent once in a while for some episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        if i % 10 == 0 or i >= (num_episodes - 1):
            evaluation_stats = []
            for j in range(100):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=False)
                evaluation_stats.append(eval_stats.episode_reward)
            mean_reward = np.mean(evaluation_stats[:5])

            solved_reward = np.mean(evaluation_stats)

            if args.verbose:
                print("evaluation: mean_reward after eposide {}:  \t{}".format(i,mean_reward))
                print("            solved_reward after eposide {}:\t{}\n".format(i,solved_reward))

            solved = solved_reward >= 195

        # store model in the end.
        if i >= (num_episodes - 1):
            save_agent(agent, model_dir, "full_run")

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "a_0" : stats.get_action_usage(0),
                                                      "a_1" : stats.get_action_usage(1),
                                                      "eval_reward" : mean_reward
                                                     })

        if solved and first_solved:
            first_solved = False
            save_agent(agent, model_dir, "solved")
            print("Task is considered as solved after {} episodes.\n\n\n".format(i))
            if not args.fullrun:
                break

    if not solved:
        print("The task has not been solved!\n\n\n")
    tensorboard.close_session()

def save_agent(agent, model_dir, model_name="dqn_agent"):
    print("Saving agent...")
    agent.saver.save(agent.sess, os.path.join(model_dir, model_name + ".ckpt"))
    print("... Agent saved!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", default=False, help="Print episode statistics.")
    parser.add_argument("--fullrun", action="store_true", default=False, help="Run until episode limit.")
    args = parser.parse_args()

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    game_name = "CartPole-v0"
    print("Starting {} ...  ".format(game_name))

    env = gym.make(game_name).unwrapped

    # 1. init Q network and target network (see dqn/networks.py)
    num_states = 4
    num_actions = env.action_space.n
    num_episodes = 500

    q_net = NeuralNetwork(num_states, num_actions, lr = 0.001)
    target_net = TargetNetwork(num_states, num_actions, lr = 0.001)

    # 2. init DQNAgent (see dqn/dqn_agent.py)
    dqn_agent = DQNAgent(game_name, q_net, target_net, num_actions)

    # 3. train DQN agent with train_online(...)
    train_online(env, dqn_agent, num_episodes)
