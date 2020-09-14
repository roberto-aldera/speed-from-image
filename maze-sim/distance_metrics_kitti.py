from pathlib import Path
import numpy as np
import pdb

from matplotlib import pyplot as plt

LENGTHS = [2, 4, 6, 8]


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
        print("Length segment:", length)
        err_tr_tmp = []
        err_d_tmp = []
        for i in range(len(gt_poses)):
            j = last_frame_from_len(dists[i:], length)
            if j is None:
                break
            j += i

            print("i:", i, "j:", j)

            se2_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[j]
            se2_est = np.linalg.inv(est_poses[i]) @ est_poses[j]
            error_se2 = np.linalg.inv(se2_est) @ se2_gt

            err_tr_tmp.append(np.sqrt(error_se2[0, 1] ** 2 + error_se2[0, 2] ** 2))
            err_d_tmp.append(abs(np.arctan2(error_se2[0, 1], error_se2[0, 0])))
        print("err_tr_tmp:", err_tr_tmp)
        print("err_d_tmp:", err_d_tmp)
        error_tr.append(np.mean(err_tr_tmp) / length)
        error_d.append(np.mean(err_d_tmp) / length)
    return error_tr, error_d


def main():
    errors_t = []
    errors_d = []

    data_folder = "/workspace/Desktop/"
    gt_df = np.genfromtxt(data_folder + "speed_labels_training_0.csv", delimiter=",")
    est_df = np.genfromtxt(data_folder + "speed_labels_training_1.csv", delimiter=",")

    # print("gt_df:", gt_df)
    err_t, err_d = get_errors(gt_df, est_df)
    errors_t.append(err_t)
    errors_d.append(err_d)

    err_t_avg = np.array(err_t).mean()
    err_d_avg = np.array(err_d).mean()
    print("err_t_avg:", err_t_avg)
    print("err_d_avg:", err_d_avg)


if __name__ == '__main__':
    main()
