import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pdf2image
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--validation_path', type=str, default="", help='path to validation folder')
    parser.add_argument('--output_path', type=str, default="maze_outputs/", help='path to output file')
    params = parser.parse_args()
    in_path = params.validation_path + "validation_maze_"
    out_path = params.output_path
    Path(out_path).mkdir(parents=True, exist_ok=True)
    print("Saving pose trajectories figure to:", out_path)

    fig = plt.figure(figsize=(45, 30))
    columns = 3
    rows = 2
    for i in range(1, columns * rows + 1):
        instance_index = i - 1
        print("Processing index:", instance_index)
        img = pdf2image.convert_from_path("%s%i%s" % (in_path, instance_index, ".pdf"))
        fig.add_subplot(rows, columns, i)
        plt.axis("off")
        plt.imshow(np.asarray(img[0]))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("%s%s%s" % (in_path, "validation_mazes", ".png"), bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    main()
