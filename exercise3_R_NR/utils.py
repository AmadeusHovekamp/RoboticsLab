import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4
ACC_LEFT = 5
ACC_RIGHT = 6
BRAKE_LEFT = 7
BRAKE_RIGHT = 8

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


def action_to_id(a):
    """
    this method discretizes the actions.
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    elif all(a == [-1.0, 1.0, 0.0]): return ACC_LEFT         # ACC_LEFT: 5
    elif all(a == [1.0, 1.0, 0.0]): return ACC_RIGHT         # ACC_RIGHT: 6
    elif all(a == [-1.0, 0.0, 0.2]): return BRAKE_LEFT       # BRAKE_LEFT: 7
    elif all(a == [1.0, 0.0, 0.2]): return BRAKE_RIGHT       # BRAKE_RIGHT: 8
    else:
        return STRAIGHT                                      # STRAIGHT = 0


def id_to_action(a):
    """
    this method undoes action_to_id.
    """
    if LEFT: return [-1.0, 0.0, 0.0]                         # LEFT: 1
    elif RIGHT: return [1.0, 0.0, 0.0]                       # RIGHT: 2
    elif ACCELERATE: return [0.0, 1.0, 0.0]                  # ACCELERATE: 3
    elif BRAKE: return [0.0, 0.0, 0.2]                       # BRAKE: 4
    elif ACC_LEFT: return [-1.0, 1.0, 0.0]                   # ACC_LEFT: 5
    elif ACC_RIGHT: return [1.0, 1.0, 0.0]                   # ACC_RIGHT: 6
    elif BRAKE_LEFT: return [-1.0, 0.0, 0.2]                 # BRAKE_LEFT: 7
    elif BRAKE_RIGHT: return [1.0, 0.0, 0.2]                 # BRAKE_RIGHT: 8
    else:
        return [0.0,0.2,0.0]                                 # STRAIGHT = 0
