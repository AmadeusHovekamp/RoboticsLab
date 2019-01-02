import numpy as np


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))

    def __str__(self):
        return "reward: {:4d}\t0: {:.4f}\t1: {:.4f}".format(int(self.episode_reward), self.get_action_usage(0), self.get_action_usage(1))
