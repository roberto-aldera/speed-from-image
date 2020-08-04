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
    start = np.array([0, int(settings.SIM_IMAGE_DIMENSION / 2)])
    w, h = settings.SIM_IMAGE_DIMENSION, settings.SIM_IMAGE_DIMENSION
    speed_labels = []
    num_samples = int(settings.TOTAL_SAMPLES * data_ratio)
    split_data_folder = "%s%s%s" % (settings.SIM_IMAGE_DIR, data_subset_type, "/")
    Path(split_data_folder).mkdir(parents=True, exist_ok=True)
    for idx in range(num_samples):
        width = np.zeros(settings.SIM_IMAGE_DIMENSION, dtype=np.uint8)
        data = np.zeros((h, w), dtype=np.uint8)
        wall_brightness = randint(150, 255)  # vary the wall brightness (but keep same for whole image)

        # Add some Gaussian noise to background of toy images
        row, col = data.shape
        mean, var = 50, 300
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col).astype(np.uint8)
        data += gauss

        step_vals = list(decomposition(settings.SIM_IMAGE_DIMENSION, settings.MIN_LENGTH,
                                       settings.MAX_LENGTH))  # generate random partitions
        total = 0

        for i in range(len(step_vals)):
            width[total:total + step_vals[i]] = randrange(settings.MIN_WIDTH, settings.MAX_WIDTH)
            total += step_vals[i]

        for i in range(settings.SIM_IMAGE_DIMENSION):
            half_wall_width = int(settings.WALL_WIDTH / 2)
            # left wall
            data[i, start[1] - width[i] - half_wall_width:start[1] - width[i] + half_wall_width] = wall_brightness
            # right wall
            data[i, start[1] + width[i] - half_wall_width:start[1] + width[i] + half_wall_width] = wall_brightness

        # Draw robot position
        robot_x = int(settings.SIM_IMAGE_DIMENSION / 2)
        robot_y = int(settings.SIM_IMAGE_DIMENSION / 2)
        data[robot_x - settings.SIM_ROBOT_RADIUS:robot_x + settings.SIM_ROBOT_RADIUS,
        robot_y - settings.SIM_ROBOT_RADIUS:robot_y + settings.SIM_ROBOT_RADIUS] = 255

        img = Image.fromarray(data, 'L')
        img.save("%s%s%s%i%s" % (split_data_folder, data_subset_type, "_", idx, ".png"))

        # Add noise to width data and treat this as speed
        speed = np.zeros(settings.SIM_IMAGE_DIMENSION - robot_x - settings.SIM_HORIZON_LENGTH)
        for i in range(len(speed)):
            speed[i] = ((width[robot_x + i]) * 2 / 3
                        + (width[robot_x + i + settings.SIM_HORIZON_LENGTH]) * 1 / 3) / 10
            speed[i] += uniform(-settings.SPEED_NOISE_LEVEL, settings.SPEED_NOISE_LEVEL)
        speed_labels.append(speed)

    np.savetxt(("%s%s%s" % (split_data_folder, data_subset_type, "_speed_labels.csv")), speed_labels, delimiter=",",
               fmt="%10.5f")
    print("Generated", num_samples, data_subset_type, "samples, with dim =", settings.SIM_IMAGE_DIMENSION,
          "and written to:", split_data_folder)

    speed_mean = np.mean(np.array(speed_labels))
    speed_std_dev = np.std(np.array(speed_labels))
    print("Speed mean for", data_subset_type, "->", speed_mean)
    print("Speed std dev for", data_subset_type, "->", speed_std_dev)


generate_and_save_samples(settings.TRAIN_RATIO, settings.TRAIN_SUBSET)
generate_and_save_samples(settings.VAL_RATIO, settings.VAL_SUBSET)
generate_and_save_samples(settings.TEST_RATIO, settings.TEST_SUBSET)

print("--- Dataset generation execution time: %s seconds ---" % (time.time() - start_time))
