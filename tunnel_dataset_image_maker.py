import numpy as np
from PIL import Image
from random import randint, randrange, uniform

n_samples = 5
dim = 28
min_width, max_width = 2, 8
wall_width = 2
path_to_store_dataset = '/Users/roberto/code/speed-from-image/images/'


def decomposition(leftovers, min_val, max_val):
    """
    Generate a sequence of random integers that are suitable for building tunnels of varying width. Must be within
    a range so that sections of the tunnel are not too short or too long (adequate variation).
    :param leftovers: the initial length of the sequence to chop into random chunks
    :param min_val: minimum random length
    :param max_val: maximum random length

    Example usage:
    tmp = list(decomposition(128, 10, 30))
    print(tmp)
    np.sum(tmp) # this should sum to whatever the original sequence length was (leftovers)
    """
    n = leftovers
    while leftovers > 0:
        if leftovers > min_val:
            n = randint(min_val, max_val)
            # avoid chance of small future value by discarding n if the new remains are too small
            if (leftovers - n) < min_val:
                n = leftovers
        yield n
        leftovers -= n


start = np.array([0, int(dim / 2)])
w, h = dim, dim
speed_labels = []

for idx in range(n_samples):
    width = np.zeros(dim, dtype=np.uint8)
    data = np.zeros((h, w), dtype=np.uint8)
    step_vals = list(decomposition(dim, min_width, max_width))  # generate random partitions
    total = 0

    for i in range(len(step_vals)):
        width[total:total + step_vals[i]] = randrange(min_width, max_width)
        total += step_vals[i]

    for i in range(dim):
        data[i, start[1] + width[i]:start[1] + width[i] + wall_width] = 255  # left wall
        data[i, start[1] - width[i]:start[1] - width[i] + wall_width] = 255  # right wall

    img = Image.fromarray(data, 'L')
    img.save("%s%s%i%s" % (path_to_store_dataset, 'img-', idx, '.png'))

    # Add noise to width data and treat this as speed
    speed = np.zeros(dim)
    for i in range(len(speed)):
        speed[i] = width[i] + uniform(-0.5, 0.5)
    speed_labels.append(speed)

np.savetxt(("%s%s" % (path_to_store_dataset, 'speed_labels.csv')), speed_labels, delimiter=',', fmt='%10.5f')
