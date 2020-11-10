import numpy as np
import pandas as pd
from argparse import ArgumentParser
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
    for i in range(len(poses)):
        p = poses[i, :]
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

    for i in range(len(ground_truth_poses)):
        se2 = gt_poses[-1] @ compose_se2(ground_truth_poses[i, 0], ground_truth_poses[i, 1], ground_truth_poses[i, 2])
        gt_poses.append(se2)

    for i in range(len(estimated_poses)):
        se2 = est_poses[-1] @ compose_se2(estimated_poses[i, 0], estimated_poses[i, 1], estimated_poses[i, 2])
        est_poses.append(se2)

    for length in segment_lengths:
        print("Evaluating on segment of length:", length)
        trans_error_per_segment = []
        yaw_error_per_segment = []
        for start_idx in range(len(gt_poses)):
            end_idx = last_frame_from_len(dists[start_idx:], length)
            if end_idx is None:
                print("There are no segments of length:", length, "so inserting NaN and moving on...")
                trans_error_per_segment.append(np.nan)
                yaw_error_per_segment.append(np.nan)
                break
            end_idx += start_idx

            # print("start_idx:", start_idx, "end_idx:", end_idx)
            se2_gt = np.linalg.inv(gt_poses[start_idx]) @ gt_poses[end_idx]
            se2_est = np.linalg.inv(est_poses[start_idx]) @ est_poses[end_idx]
            error_se2 = np.linalg.inv(se2_est) @ se2_gt
            trans_error_per_segment.append(np.sqrt(error_se2[0, 2] ** 2 + error_se2[1, 2] ** 2))
            yaw_error_per_segment.append(abs(np.arctan2(error_se2[0, 1], error_se2[0, 0])))

        print("trans_error_per_segment:", np.array(trans_error_per_segment))
        print("yaw_error_per_segment:", np.array(yaw_error_per_segment))

        translational_error.append(np.nanmean(trans_error_per_segment) / length)
        yaw_error.append(np.nanmean(yaw_error_per_segment) / length)
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
        trajectory_length = 10
        x_vals = np.repeat(1, trajectory_length)
        y_vals = np.repeat(0.1, trajectory_length)
        th_vals = np.repeat(0.01, trajectory_length)
        ground_truth_poses = np.array([x_vals, y_vals, th_vals])
        print(ground_truth_poses)
        estimated_poses = np.array(ground_truth_poses)
        estimated_poses[0, :] += 0.01  # np.random.uniform(-0.1, 0.1, trajectory_length)  # 0.05
        estimated_poses[2, 0] += 1.5
        estimated_poses[1, :] += 0.01
        estimated_poses[2, :] += 0.01  # np.random.uniform(-0.2, 0.2, trajectory_length)
        print(estimated_poses)

        from export_results import get_global_poses
        from maze_image_maker import draw_robot_poses
        x_robot, y_robot, th_robot = get_global_poses([0, 0], ground_truth_poses)
        predicted_x_robot, predicted_y_robot, predicted_th_robot = get_global_poses([0, 0], estimated_poses)
        plt.figure(figsize=(10, 10))
        draw_robot_poses(x_robot, y_robot, th_robot, "blue")
        draw_robot_poses(predicted_x_robot, predicted_y_robot, predicted_th_robot, "red")
        plt.grid()
        limit = 10 + 5
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
        plt.title("Toy example")
        plt.savefig("/workspace/Desktop/toy_example.png")
        plt.close()

    return ground_truth_poses, estimated_poses


def calculate_error_metrics(ground_truth_poses, estimated_poses, segment_lengths, decimal_precision):
    trans_error_per_length, yaw_error_per_length = get_errors(ground_truth_poses, estimated_poses, segment_lengths)
    print("----------------------------------------------------------------------------------------------------")
    print("Translation lengths:", segment_lengths)
    print("trans_error_per_length:", np.array(trans_error_per_length))
    print("yaw_error_per_length:", np.array(yaw_error_per_length))

    average_translational_error = np.nanmean(np.array(trans_error_per_length))
    average_yaw_error = np.nanmean(np.array(yaw_error_per_length))
    print("average_translational_error:", np.round(average_translational_error, decimal_precision))
    print("average_yaw_error:", np.round(average_yaw_error, decimal_precision))
    return trans_error_per_length, yaw_error_per_length


def generate_error_metrics_plots(params, segment_lengths, data_subset_type, translational_errors, yaw_errors):
    yaw_errors = np.array(yaw_errors) * 180 / np.pi
    translational_errors = np.array(translational_errors) * 100  # convert to percentage
    mean_translational_error = np.nanmean(translational_errors)
    mean_yaw_error = np.nanmean(yaw_errors)
    print("Errors for", data_subset_type, "subset:")
    print("Mean translational error (%) = ", mean_translational_error)
    print("Mean yaw error (deg/m) = ", mean_yaw_error)

    plt.figure(figsize=(15, 5))
    for i in range(params.num_samples):
        plt.plot(segment_lengths, translational_errors[i], 'o-')
    plt.xlabel("Segment Length (units)")
    plt.ylabel("Translational error (%)")
    plt.title("%s%s%s" % ("Translational error on ", data_subset_type, " set examples"))
    plt.grid()
    plt.savefig(params.validation_path + data_subset_type + "/" + data_subset_type + "_trans_errors.png")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    bar_width = 0.25
    colour_translation = 'tab:red'
    ax1.set_xlabel("Segment Length (units)")
    ax1.set_ylabel("Mean segment translational error (%)", color=colour_translation)
    ax1.bar(np.array(segment_lengths) - bar_width / 2, np.nanmean(translational_errors, axis=0), width=bar_width,
            label="Translation", color=colour_translation, alpha=0.9, zorder=3)
    ax1.tick_params(axis='y', labelcolor=colour_translation)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    colour_yaw = 'tab:blue'
    ax2.set_ylabel("Mean segment yaw error (deg/m)",
                   color=colour_yaw)  # we already handled the x-label with ax1
    ax2.bar(np.array(segment_lengths) + bar_width / 2, np.nanmean(yaw_errors, axis=0), width=bar_width, label="Yaw",
            color=colour_yaw, alpha=0.9, zorder=3)
    ax2.tick_params(axis='y', labelcolor=colour_yaw)
    ax1.grid()

    ax1.axhline(y=np.nanmean(translational_errors), color=colour_translation, linestyle='--')
    ax2.axhline(y=np.nanmean(yaw_errors), color=colour_yaw, linestyle='--')
    ax1.text(0.5, 0.95, "mean translational error (%) = " + str(np.round(mean_translational_error, 3)), ha='center',
             va='center', color=colour_translation, transform=ax1.transAxes, backgroundcolor="w")
    ax2.text(0.5, 0.9, "mean yaw error (deg/m) = " + str(np.round(mean_yaw_error, 3)), ha='center', va='center',
             color=colour_yaw, transform=ax2.transAxes, backgroundcolor="w")

    plt.title("%s%s%s" % ("Performance on ", data_subset_type, " set examples"))
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    fig.tight_layout()
    plt.savefig(params.validation_path + data_subset_type + "/" + data_subset_type + "_mean_errors.png")
    plt.close()


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


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--validation_path', type=str, default="", help='path to validation folder')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of scenarios on which to run metrics')

    params = parser.parse_args()
    print("Saving pose trajectories metrics to:", params.validation_path)

    segment_lengths = [1, 2, 3, 4, 5, 6, 7]
    decimal_precision = 3
    np.set_printoptions(precision=decimal_precision)

    calculate_and_export_distance_metrics(params, settings.TRAIN_SUBSET, segment_lengths, decimal_precision)
    calculate_and_export_distance_metrics(params, settings.VAL_SUBSET, segment_lengths, decimal_precision)
    calculate_and_export_distance_metrics(params, settings.TEST_SUBSET, segment_lengths, decimal_precision)


if __name__ == '__main__':
    main()
