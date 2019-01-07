import numpy as np
import matplotlib.pyplot as plt
import matplotlib

STRAIGHT = 0
LEFT = 1
RIGHT = 2
ACCELERATE = 3
BRAKE = 4

def one_hot(labels, classes=None):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    if classes is None:
        classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')

def get_action_name(a):
    if a == LEFT: return "LEFT"
    elif a == RIGHT: return "RIGHT"
    elif a == ACCELERATE: return "ACCELERATE"
    elif a == BRAKE: return "BRAKE"
    else:
        return "STRAIGHT"

def action_to_id(a):
    """
    this method discretizes the actions.
    """
    if all(a == np.array([-1.0, 0.0, 0.0]).astype('float32')): return LEFT               # LEFT: 1
    elif all(a == np.array([1.0, 0.0, 0.0]).astype('float32')): return RIGHT             # RIGHT: 2
    elif all(a == np.array([0.0, 1.0, 0.0]).astype('float32')): return ACCELERATE        # ACCELERATE: 3
    elif all(a == np.array([0.0, 0.0, 0.2]).astype('float32')): return BRAKE             # BRAKE: 4
    else:
        return STRAIGHT                                      # STRAIGHT = 0

def id_to_action(a):
    """
    this method undoes action_to_id.
    """
    if a == LEFT: return [-1.0, 0.00, 0.0]                         # LEFT: 1
    elif a == RIGHT: return [1.0, 0.00, 0.0]                       # RIGHT: 2
    elif a == ACCELERATE: return [0.0, 1.0, 0.0]                  # ACCELERATE: 3
    elif a == BRAKE: return [0.0, 0.0, 0.2]                       # BRAKE: 4
    else:
        return [0.0,0.0,0.0]                                 # STRAIGHT = 0

def display_state(state):
    # plt.imshow(state, cmap="rbg")
    # plt.show()
    plt.imshow(state, cmap="gray")
    plt.show()
    # plt.save("")
    pass
