import numpy as np
from PIL import Image
from random import randint, randrange, uniform
from pathlib import Path
import time
import settings

start_time = time.time()


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


def generate_and_save_samples(data_ratio, data_subset_type):
    start = np.array([0, int(settings.IMAGE_DIMENSION / 2)])
    w, h = settings.IMAGE_DIMENSION, settings.IMAGE_DIMENSION
    speed_labels = []
    num_samples = int(settings.TOTAL_SAMPLES * data_ratio)
    split_data_folder = "%s%s%s" % (settings.IMAGE_DIR, data_subset_type, "/")
    Path(split_data_folder).mkdir(parents=True, exist_ok=True)
    for idx in range(num_samples):
        width = np.zeros(settings.IMAGE_DIMENSION, dtype=np.uint8)
        data = np.zeros((h, w), dtype=np.uint8)
        wall_brightness = randint(150, 255)  # vary the wall brightness (but keep same for whole image)

        # Add some Gaussian noise to background of toy images
        row, col = data.shape
        mean, var = 50, 300
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col).astype(np.uint8)
        data += gauss

        step_vals = list(decomposition(settings.IMAGE_DIMENSION, settings.MIN_LENGTH,
                                       settings.MAX_LENGTH))  # generate random partitions
        total = 0

        for i in range(len(step_vals)):
            width[total:total + step_vals[i]] = randrange(settings.MIN_WIDTH, settings.MAX_WIDTH)
            total += step_vals[i]

        for i in range(settings.IMAGE_DIMENSION):
            # left wall
            data[i, start[1] + width[i]:start[1] + width[i] + settings.WALL_WIDTH] = wall_brightness
            # right wall
            data[i, start[1] - width[i]:start[1] - width[i] + settings.WALL_WIDTH] = wall_brightness

        img = Image.fromarray(data, 'L')
        img.save("%s%s%s%i%s" % (split_data_folder, data_subset_type, "_", idx, ".png"))

        # Add noise to width data and treat this as speed
        speed = np.zeros(settings.IMAGE_DIMENSION)
        for i in range(len(speed)):
            speed[i] = width[i] + uniform(-settings.SPEED_NOISE_LEVEL, settings.SPEED_NOISE_LEVEL)
        speed_labels.append(speed)

    np.savetxt(("%s%s%s" % (split_data_folder, data_subset_type, "_speed_labels.csv")), speed_labels, delimiter=",",
               fmt="%10.5f")
    print("Generated", num_samples, data_subset_type, "samples, with dim =", settings.IMAGE_DIMENSION,
          "and written to:", split_data_folder)


generate_and_save_samples(settings.TRAIN_RATIO, settings.TRAIN_SUBSET)
generate_and_save_samples(settings.VAL_RATIO, settings.VAL_SUBSET)
generate_and_save_samples(settings.TEST_RATIO, settings.TEST_SUBSET)

print("--- Dataset generation execution time: %s seconds ---" % (time.time() - start_time))
