import settings
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Running this test.")

    print("Using toy data for debugging purposes...")
    trajectory_length = 10
    x_vals = np.linspace(0, 10, 10)
    y_vals = np.linspace(0, 10, 10)
    plt.figure(figsize=(10, 10))
    plt.plot(x_vals, y_vals)
    plt.savefig("%s%s" % (settings.MAZE_RESULTS_DIR, "tmp.png"))
    plt.close()


def playground():
    print("Playground script...")
    my_array = np.array([0, 1, 2, 3, 4, np.NaN])
    array_mean = np.nanmean(my_array)
    print("Array is:", my_array)
    print("Array mean:", array_mean)
    print("Finished!")


if __name__ == "__main__":
    # main()
    playground()