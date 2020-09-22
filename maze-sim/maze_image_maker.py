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
    robot_th = np.array([np.pi / 2])  # robot is facing 'upwards' when moving forwards
    # Generate obstacles in random positions across the map
    obstacles_x = [random.randrange(settings.MAX_OBSTACLE_X_POSITION_FROM_CENTRE,
                                    (settings.MAP_SIZE - 1) - settings.MAX_OBSTACLE_X_POSITION_FROM_CENTRE,
                                    settings.MIN_DISTANCE_BETWEEN_OBSTACLES) for x in range(settings.MAX_NUM_OBSTACLES)]
    obstacles_y = [random.randrange(settings.MIN_OBSTACLE_Y_POSITION,
                                    (settings.MAP_SIZE - 1) - 5,
                                    settings.MIN_DISTANCE_BETWEEN_OBSTACLES) for x in range(settings.MAX_NUM_OBSTACLES)]
    obstacles = [obstacles_x, obstacles_y]
    # Remove any obstacles that are too close to the robot's starting position
    relative_positions = obstacles - np.tile(x_start, (1, settings.MAX_NUM_OBSTACLES))
    distances = np.sqrt(np.sum(np.square(relative_positions), axis=0))
    obstacles = np.delete(obstacles, np.argwhere(distances < settings.EXCLUSION_ZONE_RADIUS), axis=1)
    # Remove obstacles directly ahead of starting position
    obstacles = np.delete(obstacles, np.argwhere(obstacles[0, :] == x_start[0]), axis=1)

    # CUSTOM OBSTACLES FOR DEBUGGING ONLY
    # obstacles = np.array([[15, 14, 15, 16], [18,  21, 20, 19]])
    # obstacles = np.array([[14, 15, 16, 17], [20, 20, 20, 20]])
    # obstacles = np.array([[12, 13, 14, 15], [20, 20, 20, 20]])
    # obstacles = np.append(obstacles, np.array([[17, 18, 19, 20], [23, 23, 23, 23]]), axis=1)

    num_obstacles = obstacles.shape[1]  # account for the loss of the obstacles that were too close
    # Export obstacle location to disk
    np.savetxt(("%s%s%s%s%s%s" % (split_data_path, "/obstacles_", data_subset_type, "_", idx, ".csv")),
               obstacles, delimiter=",")

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

            # Force due to obstacle "visibility" (high if in front of robot, low if to the side or behind)
            robot_angle_from_vertical = (robot_th[-1] - robot_th[0])
            relative_angles = -np.arctan2(relative_positions[0, idx_proximal],
                                          relative_positions[1, idx_proximal]) - robot_angle_from_vertical
            relative_angles[relative_angles > np.pi] = relative_angles[relative_angles > np.pi] - 2 * np.pi
            relative_angles[relative_angles < -np.pi] = 2 * np.pi - relative_angles[relative_angles < -np.pi]
            angle_influence = (np.pi - np.abs(relative_angles)) / np.pi

            # Force due to relevance in the goal quest
            relevance_to_mission = np.array([relative_positions[1, idx_proximal]])
            relevance_to_mission[relevance_to_mission < 0] = 0
            relevance_to_mission[relevance_to_mission > 0] = 1

            f_proximity *= (1 / 3 + angle_influence / 3 + relevance_to_mission / 3)
            f_objects = settings.OBSTACLE_FORCE_MULTIPLIER * np.sum(f_proximity, axis=1).reshape(-1, 1)
        else:
            f_objects = np.array([0, 0]).reshape(-1, 1)
        f_goal = settings.GOAL_FORCE_MULTIPLIER * goal_error / np.linalg.norm(goal_error)
        f_total = f_goal + f_objects
        # This takes into account the natural slow-down that should happen when the robot is near obstacles (caution)
        max_velocity = min(settings.VELOCITY_LIMIT,
                           settings.NOMINAL_VELOCITY * min(distances) / settings.PROXIMITY_TO_OBSTACLE_CAUTION_FACTOR)
        f_total = f_total / np.linalg.norm(f_total) * min(max_velocity, np.linalg.norm(f_total))

        theta_robot = math.atan2(f_total[1], f_total[0])
        x_robot = x_robot + f_total
        robot_xy = np.append(robot_xy, x_robot, axis=1)
        robot_th = np.append(robot_th, theta_robot)
        goal_error = x_goal - x_robot
        k += 1

    if save_plots:
        plt.figure(figsize=(10, 10))
        plt.plot(obstacles[0, :], obstacles[1, :], 'r*')
        draw_robot_poses(robot_xy[0, :], robot_xy[1, :], robot_th, "blue")
        plt.plot(x_start[0], x_start[1], 'mo')
        plt.plot(x_goal[0], x_goal[1], 'go')
        plt.grid()
        plt.xlim(0, settings.MAP_SIZE)
        plt.ylim(0, settings.MAP_SIZE)
        plt.savefig("%s%s%s%s%i%s" % (split_data_path, "/", data_subset_type, "_maze_", idx, ".pdf"))
        plt.close()
    if idx % 100 == 0:
        print("Maze sim complete for", data_subset_type, "index:", idx)

    data = np.zeros((settings.MAP_SIZE, settings.MAP_SIZE), dtype=np.uint8)
    radius = settings.ADDITIONAL_OBSTACLE_VISUAL_WEIGHT
    for i in range(num_obstacles):
        data[(settings.MAP_SIZE - 1) - obstacles[1, i] - radius:
             (settings.MAP_SIZE - 1) - obstacles[1, i] + radius + 1,
        obstacles[0, i] - radius:obstacles[0, i] + radius + 1] = 255

    # Draw robot position
    # robot_x = int(settings.MAZE_IMAGE_DIMENSION / 2)
    # robot_y = int(settings.MAZE_IMAGE_DIMENSION / 2)
    # data[robot_x - settings.ADDITIONAL_ROBOT_VISUAL_WEIGHT:robot_x + settings.ADDITIONAL_ROBOT_VISUAL_WEIGHT + 1,
    # robot_y - settings.ADDITIONAL_ROBOT_VISUAL_WEIGHT:robot_y + settings.ADDITIONAL_ROBOT_VISUAL_WEIGHT + 1] = 255

    img = Image.fromarray(data, 'L')
    img.save("%s%s%s%s%i%s" % (split_data_path, "/", data_subset_type, "_", idx, ".png"))

    return robot_xy, robot_th


def draw_robot_poses(x_poses, y_poses, thetas, colour):
    scale = 0.25
    basic_triangle = np.array([[-1, -1], [0, 2], [1, -1]]) * scale

    for i in range(len(x_poses)):
        th = -(thetas[i] - np.pi / 2)
        rotation_matrix = np.array([[-np.cos(th), np.sin(th)], [np.sin(th), np.cos(th)]])
        triangle_vertices = np.matmul(basic_triangle, rotation_matrix)
        triangle_vertices[:, 0] = triangle_vertices[:, 0] + x_poses[i]
        triangle_vertices[:, 1] = triangle_vertices[:, 1] + y_poses[i]
        triangle = plt.Polygon(triangle_vertices, color=colour, alpha=1.0, fill=False)
        plt.gca().plot(x_poses[i], y_poses[i], '.', color=colour, alpha=1.0)
        plt.gca().add_patch(triangle)


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


def generate_maze_samples(num_samples, data_subset_type):
    split_data_path = Path(settings.MAZE_IMAGE_DIR, data_subset_type)
    if split_data_path.exists() and split_data_path.is_dir():
        shutil.rmtree(split_data_path)
    split_data_path.mkdir(parents=True)
    save_plots = False

    for idx in range(num_samples):
        xy_positions, thetas = run_maze_sim_and_generate_images(idx, split_data_path, data_subset_type,
                                                                save_plots)
        generate_relative_poses(idx, xy_positions, thetas, split_data_path, data_subset_type, save_plots)

    print("Generated", num_samples, data_subset_type, "samples, with dim =", settings.MAZE_IMAGE_DIMENSION,
          "and written to:", split_data_path)


if __name__ == "__main__":
    start_time = time.time()

    generate_maze_samples(settings.TRAIN_SET_SIZE, settings.TRAIN_SUBSET)
    generate_maze_samples(settings.VAL_SET_SIZE, settings.VAL_SUBSET)
    generate_maze_samples(settings.TEST_SET_SIZE, settings.TEST_SUBSET)
    print("--- Dataset generation execution time: %s seconds ---" % (time.time() - start_time))
