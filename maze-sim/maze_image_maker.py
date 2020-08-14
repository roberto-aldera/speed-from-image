import numpy as np
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import time
import settings


def run_maze_sim_and_generate_images(idx, split_data_path, data_subset_type, save_plots=False):
    k = 0
    x_start = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE / 2]).reshape(2, -1)
    x_goal = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE]).reshape(2, -1)
    x_robot = x_start
    goal_error = x_goal - x_robot
    robot_xy = np.array(x_robot)
    robot_th = np.array(0)
    # Generate obstacles in random positions across the map
    # obstacles = np.random.randint(0, settings.MAP_SIZE - 1, size=(2, settings.MAX_NUM_OBSTACLES))
    obstacles = [[random.randrange(0, settings.MAP_SIZE - 1, settings.MIN_DISTANCE_BETWEEN_OBSTACLES) for x in
                  range(settings.MAX_NUM_OBSTACLES)] for y in range(2)]
    # Remove any obstacles that are too close to the robot's starting position
    relative_positions = obstacles - np.tile(x_start, (1, settings.MAX_NUM_OBSTACLES))
    distances = np.sqrt(np.sum(np.square(relative_positions), axis=0))
    obstacles = np.delete(obstacles, np.argwhere(distances < settings.EXCLUSION_ZONE_RADIUS), axis=1)
    # Remove obstacles directly ahead of starting position
    obstacles = np.delete(obstacles, np.argwhere(obstacles[0, :] == x_start[0]), axis=1)
    num_obstacles = obstacles.shape[1]  # account for the loss of the obstacles that were too close

    while k < settings.MAX_ITERATIONS:
        relative_positions = obstacles - np.tile(x_robot, (1, num_obstacles))
        for i in range(relative_positions.shape[1]):
            if np.all(relative_positions[:, i] == [0, 0]):
                print("Error: obstacle and robot are both at position:", obstacles[:, i],
                      "so will now move robot slightly away from this")
                relative_positions[:, i] = [1e-3, 1e-3]
                print("k = ", k)
        distances = np.sqrt(np.sum(np.square(relative_positions), axis=0))
        idx_proximal = distances < settings.OBSTACLE_INFLUENCE_RADIUS
        if any(idx_proximal):
            rho = np.tile(distances[idx_proximal], (2, 1))
            v = relative_positions[:, idx_proximal]
            d_rho_dx = -v / rho
            f_proximity = (1 / rho - 1 / settings.OBSTACLE_INFLUENCE_RADIUS) * 1 / (np.square(rho)) * d_rho_dx
            f_objects = settings.OBSTACLE_FORCE_MULTIPLIER * np.sum(f_proximity, axis=1).reshape(-1, 1)
        else:
            f_objects = np.array([0, 0]).reshape(-1, 1)
        f_goal = settings.GOAL_FORCE_MULTIPLIER * goal_error / np.linalg.norm(goal_error)
        f_total = f_goal + f_objects
        f_total = f_total / np.linalg.norm(f_total) * min(settings.VELOCITY_LIMIT, np.linalg.norm(f_total))
        theta_robot = math.atan2(f_total[1], f_total[0])
        x_robot = x_robot + f_total
        robot_xy = np.append(robot_xy, x_robot, axis=1)
        robot_th = np.append(robot_th, theta_robot)
        goal_error = x_goal - x_robot
        k += 1

    if save_plots:
        plt.figure(figsize=(10, 10))
        plt.plot(obstacles[0, :], obstacles[1, :], 'r*')
        plt.plot(robot_xy[0, :], robot_xy[1, :], 'b^')
        plt.plot(x_start[0], x_start[1], 'mo')
        plt.plot(x_goal[0], x_goal[1], 'go')
        plt.grid()
        plt.xlim(0, settings.MAP_SIZE)
        plt.ylim(0, settings.MAP_SIZE)
        plt.savefig("%s%s%s%s%i%s" % (split_data_path, "/", data_subset_type, "_maze_", idx, ".png"))
        plt.close()
    print("Maze sim complete for index:", idx)

    data = np.zeros((settings.MAP_SIZE, settings.MAP_SIZE), dtype=np.uint8)
    radius = settings.ADDITIONAL_OBSTACLE_VISUAL_WEIGHT
    for i in range(num_obstacles):
        data[(settings.MAP_SIZE - 1) - obstacles[1, i] - radius:
             (settings.MAP_SIZE - 1) - obstacles[1, i] + radius + 1,
        obstacles[0, i] - radius:obstacles[0, i] + radius + 1] = 255

    # Draw robot position
    robot_x = int(settings.MAZE_IMAGE_DIMENSION / 2)
    robot_y = int(settings.MAZE_IMAGE_DIMENSION / 2)
    data[robot_x - settings.ADDITIONAL_ROBOT_VISUAL_WEIGHT:robot_x + settings.ADDITIONAL_ROBOT_VISUAL_WEIGHT + 1,
    robot_y - settings.ADDITIONAL_ROBOT_VISUAL_WEIGHT:robot_y + settings.ADDITIONAL_ROBOT_VISUAL_WEIGHT + 1] = 255

    img = Image.fromarray(data, 'L')
    img.save("%s%s%s%s%i%s" % (split_data_path, "/", data_subset_type, "_", idx, ".png"))

    return robot_xy, robot_th


def generate_relative_poses(idx, robot_xy, robot_th, split_data_path, data_subset_type, save_plots=False):
    x_start = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE / 2]).reshape(2, -1)
    T_robot_world = np.identity(3)
    th = np.pi / 2  # facing upwards
    T_robot_world[0, 0] = np.cos(th)
    T_robot_world[0, 1] = -np.sin(th)
    T_robot_world[1, 0] = np.sin(th)
    T_robot_world[1, 1] = np.cos(th)
    T_robot_world[0, 2] = x_start[0]
    T_robot_world[1, 2] = x_start[1]
    robot_global_poses = [T_robot_world]
    relative_poses = []

    for i in range(1, robot_xy.shape[1]):
        T_i = np.identity(3)
        th = robot_th[i]
        T_i[0, 0] = np.cos(th)
        T_i[0, 1] = -np.sin(th)
        T_i[1, 0] = np.sin(th)
        T_i[1, 1] = np.cos(th)
        T_i[0, 2] = robot_xy[:, i][0]
        T_i[1, 2] = robot_xy[:, i][1]
        robot_global_poses.append(T_i)

    for i in range(1, len(robot_global_poses)):
        T_rel_pose = np.matmul(np.linalg.inv(robot_global_poses[i - 1]), robot_global_poses[i])
        relative_poses.append(T_rel_pose)

    dx = []
    dy = []
    dth = []
    for i in range(len(relative_poses)):
        dx.append(relative_poses[i][0, 2])
        dy.append(relative_poses[i][1, 2])
        dth.append(np.arctan2(relative_poses[i][1, 0], relative_poses[i][1, 1]))

    np.savetxt(("%s%s%s%s%s%s" % (split_data_path, "/speed_labels_", data_subset_type, "_", idx, ".csv")),
               (dx, dy, dth), delimiter=",",
               fmt="%10.5f")

    if save_plots:
        plt.figure(figsize=(10, 3))
        plt.plot(dx, '.-', label="x")
        plt.plot(dy, '.-', label="y")
        plt.plot(dth, '.-', label="yaw")
        plt.grid()
        plt.legend()
        plt.savefig("%s%s%i%s" % (split_data_path, "/dx_dy_dth_", idx, ".png"))
        plt.close()


def generate_maze_samples(data_ratio, data_subset_type):
    split_data_path = Path(settings.MAZE_IMAGE_DIR, data_subset_type)
    if split_data_path.exists() and split_data_path.is_dir():
        shutil.rmtree(split_data_path)
    split_data_path.mkdir(parents=True)
    num_samples = int(settings.TOTAL_SAMPLES * data_ratio)
    save_plots = True

    for idx in range(num_samples):
        xy_positions, thetas = run_maze_sim_and_generate_images(idx, split_data_path, data_subset_type,
                                                                save_plots)
        generate_relative_poses(idx, xy_positions, thetas, split_data_path, data_subset_type, save_plots)

    print("Generated", num_samples, data_subset_type, "samples, with dim =", settings.MAZE_IMAGE_DIMENSION,
          "and written to:", split_data_path)


start_time = time.time()

generate_maze_samples(settings.TRAIN_RATIO, settings.TRAIN_SUBSET)
# generate_maze_samples(settings.VAL_RATIO, settings.VAL_SUBSET)
# generate_maze_samples(settings.TEST_RATIO, settings.TEST_SUBSET)
print("--- Dataset generation execution time: %s seconds ---" % (time.time() - start_time))
