import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import settings


def compose_se2(x, y, angle):
    se2 = np.zeros((3, 3))
    se2[0, 2] = x
    se2[1, 2] = y
    se2[0, 0] = np.cos(angle)
    se2[0, 1] = np.sin(angle)
    se2[1, 0] = -np.sin(angle)
    se2[1, 1] = np.cos(angle)
    se2[2, 2] = 1
    return se2


def trajectory_distances(poses):
    dist = [0.]
    for i in range(0, poses.shape[1]):
        p = poses[:, i]
        dx = p[0]
        dy = p[1]

        dist.append(dist[-1] + np.sqrt(dx * dx + dy * dy))
    print("dist:", np.array(dist))
    return dist


def last_frame_from_len(dist, length):
    for i, d in enumerate(dist):
        if d >= dist[0] + length:
            return i
    return None


def get_errors(ground_truth_poses, estimated_poses, segment_lengths):
    translational_error = []
    yaw_error = []
    dists = trajectory_distances(ground_truth_poses)
    gt_poses = [np.eye(3)]
    est_poses = [np.eye(3)]

    for i in range(0, ground_truth_poses.shape[1]):
        se2 = gt_poses[-1] @ compose_se2(ground_truth_poses[0, i], ground_truth_poses[1, i], ground_truth_poses[2, i])
        gt_poses.append(se2)

    for i in range(0, estimated_poses.shape[1]):
        se2 = est_poses[-1] @ compose_se2(estimated_poses[0, i], estimated_poses[1, i], estimated_poses[2, i])
        est_poses.append(se2)

    for length in segment_lengths:
        print("Evaluating on segment of length:", length)
        trans_error_per_segment = []
        yaw_error_per_segment = []
        for start_idx in range(len(gt_poses)):
            end_idx = last_frame_from_len(dists[start_idx:], length)
            if end_idx is None:
                break
            end_idx += start_idx

            print("start_idx:", start_idx, "end_idx:", end_idx)
            se2_gt = np.linalg.inv(gt_poses[start_idx]) @ gt_poses[end_idx]
            se2_est = np.linalg.inv(est_poses[start_idx]) @ est_poses[end_idx]
            error_se2 = np.linalg.inv(se2_est) @ se2_gt
            trans_error_per_segment.append(np.sqrt(error_se2[0, 2] ** 2 + error_se2[1, 2] ** 2))
            yaw_error_per_segment.append(abs(np.arctan2(error_se2[0, 1], error_se2[0, 0])))
        if len(trans_error_per_segment) < 1:
            print("No segments of length:", length)
            return None, None
        print("trans_error_per_segment:", np.array(trans_error_per_segment))
        print("yaw_error_per_segment:", np.array(yaw_error_per_segment))

        translational_error.append(np.mean(trans_error_per_segment) / length)
        yaw_error.append(np.mean(yaw_error_per_segment) / length)
    return translational_error, yaw_error


def get_poses(folder_path, data_subset_type, idx):
    ground_truth_poses = np.genfromtxt(
        folder_path + data_subset_type + "/ground_truth_" + data_subset_type + "_" + str(idx) + ".csv",
        delimiter=",")
    estimated_poses = np.genfromtxt(
        folder_path + data_subset_type + "/prediction_" + data_subset_type + "_" + str(idx) + ".csv",
        delimiter=",")

    # Debugging with toy data
    use_toy_data = False
    if use_toy_data:
        print("Using toy data for debugging purposes...")
        x_vals = np.repeat(1, 5)
        y_vals = np.repeat(0.1, 5)
        th_vals = np.repeat(0.01, 5)
        ground_truth_poses = np.array([x_vals, y_vals, th_vals])
        print(ground_truth_poses)
        estimated_poses = np.array(ground_truth_poses)
        estimated_poses[0, :] += 0.1
        estimated_poses[1, :] += 0.05
        estimated_poses[2, :] += 0.01
        print(estimated_poses)

    return ground_truth_poses, estimated_poses


def calculate_error_metrics(ground_truth_poses, estimated_poses, segment_lengths, decimal_precision):
    trans_error_per_length, yaw_error_per_length = get_errors(ground_truth_poses, estimated_poses, segment_lengths)
    print("----------------------------------------------------------------------------------------------------")
    print("Translation lengths:", segment_lengths)

    print("trans_error_per_length:", np.array(trans_error_per_length))
    print("yaw_error_per_length:", np.array(yaw_error_per_length))

    if (trans_error_per_length or yaw_error_per_length) is None:
        print("One of the length segments requested is too long.")
        return

    average_translational_error = np.array(trans_error_per_length).mean()
    average_yaw_error = np.array(yaw_error_per_length).mean()
    print("average_translational_error:", np.round(average_translational_error, decimal_precision))
    print("average_yaw_error:", np.round(average_yaw_error, decimal_precision))
    return trans_error_per_length, yaw_error_per_length


def generate_error_metrics_plots(params, segment_lengths, data_subset_type, translational_errors, yaw_errors):
    plt.figure(figsize=(15, 5))
    for i in range(params.num_samples):
        plt.plot(segment_lengths, translational_errors[i], 'o-')
    # plt.ylim(0, 0.2)
    plt.xlabel("Segment Length (units)")
    plt.ylabel("Translational error (%)")
    plt.title("%s%s%s" % ("Translational error on ", data_subset_type, " set examples"))
    plt.grid()
    plt.savefig(params.validation_path + data_subset_type + "/" + data_subset_type + "_trans_errors.png")
    plt.close()

    plt.figure(figsize=(15, 5))
    bar_width = 0.25
    plt.bar(np.array(segment_lengths) - bar_width / 2, np.mean(translational_errors, axis=0), width=bar_width,
            label="Translation")
    plt.bar(np.array(segment_lengths) + bar_width / 2, np.mean(yaw_errors, axis=0), width=bar_width, label="Yaw")
    # plt.ylim(0, 0.2)
    plt.xlabel("Segment Length (units)")
    plt.ylabel("Mean error (%, rad/length unit)")
    plt.title("%s%s%s" % ("Performance on ", data_subset_type, " set examples"))
    plt.grid()
    plt.legend()
    plt.savefig(params.validation_path + data_subset_type + "/" + data_subset_type + "_mean_errors.png")
    plt.close()


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--validation_path', type=str, default="", help='path to validation folder')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of scenarios on which to run metrics')

    params = parser.parse_args()
    print("Saving pose trajectories metrics to:", params.validation_path)

    segment_lengths = [1, 2, 3, 4, 5, 6, 7]
    decimal_precision = 3
    np.set_printoptions(precision=decimal_precision)

    calculate_and_export_distance_metrics(params, settings.TRAIN_SUBSET, segment_lengths, decimal_precision)
    calculate_and_export_distance_metrics(params, settings.VAL_SUBSET, segment_lengths, decimal_precision)
    calculate_and_export_distance_metrics(params, settings.TEST_SUBSET, segment_lengths, decimal_precision)


def calculate_and_export_distance_metrics(params, data_subset_type, segment_lengths, decimal_precision):
    translational_errors = []
    yaw_errors = []

    for idx in range(params.num_samples):
        ground_truth_poses, estimated_poses = get_poses(params.validation_path, data_subset_type, idx)
        translational_error, yaw_error = calculate_error_metrics(ground_truth_poses, estimated_poses,
                                                                 segment_lengths, decimal_precision)

        translational_errors.append(translational_error)
        yaw_errors.append(yaw_error)
    print("----------------------------------------------------------------------------------------------------")
    trans_errors_df = pd.DataFrame(translational_errors, columns=segment_lengths)
    yaw_errors_df = pd.DataFrame(np.array(yaw_errors) * 180 / np.pi, columns=segment_lengths)
    print("Translational errors for all samples in this set:\n", trans_errors_df)
    print("Yaw errors (deg) for all samples in this set:\n", yaw_errors_df)
    trans_errors_df.to_csv(
        params.validation_path + data_subset_type + "/" + data_subset_type + "_translational_errors.csv")
    yaw_errors_df.to_csv(params.validation_path + data_subset_type + "/" + data_subset_type + "_yaw_errors.csv")

    generate_error_metrics_plots(params, segment_lengths, data_subset_type, translational_errors, yaw_errors)


if __name__ == '__main__':
    main()
