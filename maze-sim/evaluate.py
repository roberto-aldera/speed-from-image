from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
from argparse import ArgumentParser
import logging

import settings
from dataset_loader import MazeDataset, ToTensor
from lenet import LeNet
from resnet import ResNet
from maze_image_maker import draw_robot_poses

data_transform_for_evaluation = transforms.Compose([ToTensor()])


def generate_subset_evaluation_plots(data_subset_type, model, results_path, num_samples_to_eval):
    dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                          data_subset_type=data_subset_type,
                          transform=data_transform_for_evaluation)
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=1)
    subset_fig_path = results_path + data_subset_type + "/"
    Path(subset_fig_path).mkdir(parents=True, exist_ok=True)
    print("Saving evaluation plots to:", subset_fig_path)

    for i in range(num_samples_to_eval):
        img = data_loader.dataset[i]['image'].unsqueeze_(0)
        pose_labels = data_loader.dataset[i]['pose_data']
        pose_from_model = model(img).detach().numpy()
        pose_estimate = np.transpose(
            (np.transpose(pose_from_model.squeeze(0)) * settings.MAZE_SPEED_STD_DEV) + settings.MAZE_SPEED_MEAN)
        plt.figure(figsize=(15, 5))
        plt.plot(pose_labels[0], 'r--', alpha=0.5, label="dx ground truth")
        plt.plot(pose_labels[1], 'g--', alpha=0.5, label="dy ground truth")
        plt.plot(pose_labels[2], 'b--', alpha=0.5, label="dth ground truth")
        plt.plot(pose_estimate[0], 'r', label="dx prediction")
        plt.plot(pose_estimate[1], 'g', label="dy prediction")
        plt.plot(pose_estimate[2], 'b', label="dth prediction")

        plt.ylim(-1, 1)
        plt.xlabel("Index")
        plt.ylabel("dx")
        plt.title("%s%s%s" % ("Performance on ", data_subset_type, " set example"))
        plt.legend()
        plt.grid()
        plt.savefig("%s%s%s%i%s" % (subset_fig_path, data_subset_type, "-performance_", i, ".png"))
        plt.close()


def calculate_rmse(data_subset_type, model, logger):
    dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                          data_subset_type=data_subset_type,
                          transform=data_transform_for_evaluation)
    data_loader = DataLoader(dataset, batch_size=1,  # not sure if batch size here needs to be only 1
                             shuffle=False, num_workers=settings.NUM_CPUS)
    logger.info("RMSE for " + str(len(data_loader)) + " " + data_subset_type + " set examples:")
    cumulative_rmse = 0

    for i in range(len(data_loader)):
        if i % 100 == 0:
            print("Running RMSE calculation for sample index:", i)
        #     now = time.time()
        data_at_idx = data_loader.dataset[i]  # this is slow
        # t1 = time.time() - now
        img = data_at_idx['image'].unsqueeze_(0)
        # t2 = time.time() - now - t1
        pose_labels = data_at_idx['pose_data'].numpy()
        # t3 = time.time() - now - t1 - t2
        pose_from_model = model(img).detach().numpy()  # this is also slow
        # t4 = time.time() - now - t1 - t2 - t3
        pose_estimate = np.transpose(
            (np.transpose(pose_from_model.squeeze(0)) * settings.MAZE_SPEED_STD_DEV) + settings.MAZE_SPEED_MEAN)
        rmse = np.sqrt(np.mean(np.square(pose_labels - pose_estimate) / len(pose_labels), axis=1))
        cumulative_rmse += rmse
        # loop_time = time.time() - now
        # print("t1", t1)
        # print("t2", t2)
        # print("t3", t3)
        # print("t4", t4)
        # print("loop time:", loop_time)
    logger.info(cumulative_rmse / len(data_loader))


def export_figures_for_poses(results_path, data_subset_type, model, num_samples):
    dataset = MazeDataset(root_dir=settings.MAZE_IMAGE_DIR,
                          data_subset_type=data_subset_type,
                          transform=data_transform_for_evaluation)
    data_loader = DataLoader(dataset, batch_size=1,  # not sure if batch size here needs to be only 1
                             shuffle=False, num_workers=settings.NUM_CPUS)

    subset_fig_path = results_path + data_subset_type + "/"
    Path(subset_fig_path).mkdir(parents=True, exist_ok=True)
    print("Saving pose trajectory figures to:", subset_fig_path)

    x_start = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE / 2]).reshape(2, -1)
    x_goal = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE]).reshape(2, -1)

    for idx in range(num_samples):
        obstacles = np.genfromtxt(
            settings.MAZE_IMAGE_DIR + data_subset_type + "/obstacles_" + data_subset_type + "_" + str(idx) + ".csv",
            delimiter=",")

        data_at_idx = data_loader.dataset[idx]
        img = data_at_idx['image'].unsqueeze_(0)
        pose_labels = data_at_idx['pose_data'].numpy()
        pose_from_model = model(img).detach().numpy()
        pose_estimate = np.transpose(
            (np.transpose(pose_from_model.squeeze(0)) * settings.MAZE_SPEED_STD_DEV) + settings.MAZE_SPEED_MEAN)

        x_robot, y_robot, th_robot = get_global_poses(x_start, pose_labels)
        predicted_x_robot, predicted_y_robot, predicted_th_robot = get_global_poses(x_start, pose_estimate)

        plt.figure(figsize=(10, 10))
        colours = ["red", "blue", "orange", "magenta", "green"]
        plt.plot(obstacles[0, :], obstacles[1, :], "*", color=colours[0], label="obstacles")
        draw_robot_poses(x_robot, y_robot, th_robot, colours[1])
        draw_robot_poses(predicted_x_robot, predicted_y_robot, predicted_th_robot, colours[2])
        plt.plot(x_start[0], x_start[1], "o", color=colours[3])
        plt.plot(x_goal[0], x_goal[1], "o", color=colours[4])
        plt.grid()
        plt.xlim(0, settings.MAP_SIZE)
        plt.ylim(0, settings.MAP_SIZE)
        lines = [plt.Line2D([0], [0], color=c, linewidth=0, marker='.', markersize=20) for c in colours]
        labels = ["Obstacles", "True pose", "Predicted pose", "Start", "Goal"]
        plt.legend(lines, labels)
        plt.title("%s%s%s" % ("Maze simulation from ", data_subset_type, " subset"))
        plt.savefig("%s%s%s%s%i%s" % (subset_fig_path, "/", data_subset_type, "_maze_", idx, ".pdf"))
        plt.close()


def get_global_poses(x_start, relative_poses_list):
    # Make global poses from relative poses
    T_robot_start = np.identity(3)
    th = np.pi / 2  # facing upwards
    T_robot_start[0, 0] = np.cos(th)
    T_robot_start[0, 1] = -np.sin(th)
    T_robot_start[1, 0] = np.sin(th)
    T_robot_start[1, 1] = np.cos(th)
    T_robot_start[0, 2] = x_start[0]
    T_robot_start[1, 2] = x_start[1]
    robot_global_poses = [T_robot_start]
    relative_poses = []

    for i in range(1, relative_poses_list.shape[1]):
        T_i = np.identity(3)
        th = relative_poses_list[2, i]
        T_i[0, 0] = np.cos(th)
        T_i[0, 1] = -np.sin(th)
        T_i[1, 0] = np.sin(th)
        T_i[1, 1] = np.cos(th)
        T_i[0, 2] = relative_poses_list[0, i]
        T_i[1, 2] = relative_poses_list[1, i]
        relative_poses.append(T_i)

    x_robot = []
    y_robot = []
    th_robot = []
    for i in range(1, len(relative_poses)):
        T_global_pose = np.matmul((robot_global_poses[i - 1]), relative_poses[i])
        robot_global_poses.append(T_global_pose)
        x_robot.append(T_global_pose[0, 2])
        y_robot.append(T_global_pose[1, 2])
        th_robot.append(np.arctan2(T_global_pose[1, 0], T_global_pose[1, 1]))
    return x_robot, y_robot, th_robot


def do_quick_evaluation(hparams, model, model_path):
    start_time = time.time()
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    results_path = settings.MAZE_RESULTS_DIR + hparams.model_name + "/" + current_time + "/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    model = model.load_from_checkpoint(model_path)
    model.eval()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(message)s', filename=(results_path + "rmse_results.txt"), level=logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info("Loaded model from " + model_path + " -> ready to evaluate.")

    print("Generating evaluation plots...")
    num_samples = 10
    export_figures_for_poses(results_path, settings.TRAIN_SUBSET, model, num_samples)
    export_figures_for_poses(results_path, settings.VAL_SUBSET, model, num_samples)
    export_figures_for_poses(results_path, settings.TEST_SUBSET, model, num_samples)

    generate_subset_evaluation_plots(settings.TRAIN_SUBSET, model, results_path, num_samples)
    generate_subset_evaluation_plots(settings.VAL_SUBSET, model, results_path, num_samples)
    generate_subset_evaluation_plots(settings.TEST_SUBSET, model, results_path, num_samples)

    print("Calculating average RMSE (over entire subset)")

    calculate_rmse(settings.TRAIN_SUBSET, model, logger)
    calculate_rmse(settings.VAL_SUBSET, model, logger)
    calculate_rmse(settings.TEST_SUBSET, model, logger)

    print("--- Evaluation execution time: %s seconds ---" % (time.time() - start_time))


def main():
    Path(settings.MAZE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.MAZE_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default=settings.ARCHITECTURE_TYPE, help='lenet or resnet')

    temp_args, _ = parser.parse_known_args()
    # let the model add what it wants
    if temp_args.model_name == 'lenet':
        parser = LeNet.add_model_specific_args(parser)
    elif temp_args.model_name == 'resnet':
        parser = ResNet.add_model_specific_args(parser)

    hparams = parser.parse_args()
    path_to_model = "%s%s%s" % (settings.MAZE_MODEL_DIR, hparams.model_name, ".ckpt")

    # pick model
    model = None
    if hparams.model_name == 'lenet':
        model = LeNet(hparams)
    elif hparams.model_name == 'resnet':
        model = ResNet(hparams)

    do_quick_evaluation(hparams, model, path_to_model)


if __name__ == "__main__":
    main()
