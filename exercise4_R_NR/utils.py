import numpy as np
from carracing_utils import get_action_name


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
        result = "reward: {:4d}\t".format(int(self.episode_reward))

        for action in sorted(set(self.actions_ids)):
            # result += "\taction {}: {:.4f}".format(action, self.get_action_usage(action))
            result += "{}: {:.4f}".format(get_action_name(action), self.get_action_usage(action))
        return result
