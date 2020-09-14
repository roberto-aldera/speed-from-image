import numpy as np
import pdb

LENGTHS = [1, 3, 6]


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


def get_errors(gt, est):
    error_tr = []
    error_d = []

    dists = trajectory_distances(gt)

    gt_poses = [np.eye(3)]
    est_poses = [np.eye(3)]

    for i in range(0, gt.shape[1]):
        se2 = gt_poses[-1] @ compose_se2(gt[0, i], gt[1, i], gt[2, i])
        gt_poses.append(se2)

    for i in range(0, est.shape[1]):
        se2 = est_poses[-1] @ compose_se2(est[0, i], est[1, i], est[2, i])
        est_poses.append(se2)

    for length in LENGTHS:
        print("Evaluating on segment of length:", length)
        err_tr_tmp = []
        err_d_tmp = []
        for start_idx in range(len(gt_poses)):
            end_idx = last_frame_from_len(dists[start_idx:], length)
            if end_idx is None:
                break
            end_idx += start_idx

            print("start_idx:", start_idx, "end_idx:", end_idx)

            se2_gt = np.linalg.inv(gt_poses[start_idx]) @ gt_poses[end_idx]
            se2_est = np.linalg.inv(est_poses[start_idx]) @ est_poses[end_idx]
            error_se2 = np.linalg.inv(se2_est) @ se2_gt

            err_tr_tmp.append(np.sqrt(error_se2[0, 1] ** 2 + error_se2[0, 2] ** 2))
            err_d_tmp.append(abs(np.arctan2(error_se2[0, 1], error_se2[0, 0])))
        if len(err_tr_tmp) < 1:
            print("No segments of length:", length)
            return None, None  # this is not a graceful way of handling longer segments, fix this
        print("err_tr_tmp:", err_tr_tmp)
        print("err_d_tmp:", err_d_tmp)
        error_tr.append(np.mean(err_tr_tmp) / length)
        error_d.append(np.mean(err_d_tmp) / length)
    return error_tr, error_d


def main():
    errors_t = []
    errors_d = []

    # data_folder = "/workspace/Desktop/"
    # ground_truth_data = np.genfromtxt(data_folder + "speed_labels_training_0.csv", delimiter=",")
    # estimated_data = np.genfromtxt(data_folder + "speed_labels_training_1.csv", delimiter=",")

    x_vals = np.repeat(1, 5)
    y_vals = np.repeat(0.1, 5)
    th_vals = np.repeat(0.01, 5)
    ground_truth_data = np.array([x_vals, y_vals, th_vals])
    print(ground_truth_data)
    estimated_data = np.array(ground_truth_data)
    estimated_data[0, :] += 0.1
    estimated_data[1, :] += 0.05
    estimated_data[2, :] += 0.01
    print(estimated_data)

    err_t, err_d = get_errors(ground_truth_data, estimated_data)
    print(err_t)
    errors_t.append(err_t)
    errors_d.append(err_d)

    # err_t_avg = np.array(err_t).mean()
    # err_d_avg = np.array(err_d).mean()
    # print("err_t_avg:", err_t_avg)
    # print("err_d_avg:", err_d_avg)


if __name__ == '__main__':
    main()
