import numpy as np


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
    print("dist:", dist)
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

            trans_error_per_segment.append(np.sqrt(error_se2[0, 1] ** 2 + error_se2[0, 2] ** 2))
            yaw_error_per_segment.append(abs(np.arctan2(error_se2[0, 1], error_se2[0, 0])))
        if len(trans_error_per_segment) < 1:
            print("No segments of length:", length)
            return None, None
        print("trans_error_per_segment:", np.array(trans_error_per_segment))
        print("yaw_error_per_segment:", np.array(yaw_error_per_segment))

        translational_error.append(np.mean(trans_error_per_segment) / length)
        yaw_error.append(np.mean(yaw_error_per_segment) / length)
    return translational_error, yaw_error


def main():
    segment_lengths = [1, 2, 3]
    decimal_precision = 3
    np.set_printoptions(precision=decimal_precision)

    # translational_errors = []
    # yaw_errors = []

    data_folder = "/workspace/Desktop/"
    ground_truth_poses = np.genfromtxt(data_folder + "speed_labels_training_0.csv", delimiter=",")
    estimated_poses = np.genfromtxt(data_folder + "speed_labels_training_1.csv", delimiter=",")

    # Debugging with toy data
    use_toy_data = False
    if use_toy_data:
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

    trans_error_per_length, yaw_error_per_length = get_errors(ground_truth_poses, estimated_poses, segment_lengths)
    print("----------------------------------------------------------------------------------------------------")
    print("Translation lengths:", segment_lengths)

    print("trans_error_per_length:", np.array(trans_error_per_length))
    print("yaw_error_per_length:", np.array(yaw_error_per_length))

    if (trans_error_per_length or yaw_error_per_length) is None:
        print("One of the length segments requested is too long.")
        return

    # This will make more sense if we are looping over a few trajectories
    # translational_errors.append(translational_error)
    # yaw_errors.append(yaw_error)

    average_translational_error = np.array(trans_error_per_length).mean()
    average_yaw_error = np.array(yaw_error_per_length).mean()
    print("average_translational_error:", np.round(average_translational_error, decimal_precision))
    print("average_yaw_error:", np.round(average_yaw_error, decimal_precision))


if __name__ == '__main__':
    main()
