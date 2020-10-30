import numpy as np
import math
import random
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import time
import csv
import settings


def run_maze_sim_and_generate_images(idx, split_data_path, data_subset_type, save_plots=False):
    k = 0
    x_start = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE / 2])
    x_goal = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE])
    x_robot = x_start
    goal_error = x_goal - x_robot
    robot_xy = np.array(x_robot).reshape(1, -1)
    robot_th = np.array([np.pi / 2])  # robot is facing 'upwards' when moving forwards (+theta is ccw in global ref)
    # Generate obstacles in random positions across the map
    obstacles_x = [random.randrange(settings.MAX_OBSTACLE_X_POSITION_FROM_CENTRE,
                                    (settings.MAP_SIZE - 1) - settings.MAX_OBSTACLE_X_POSITION_FROM_CENTRE,
                                    settings.MIN_DISTANCE_BETWEEN_OBSTACLES) for _ in range(settings.MAX_NUM_OBSTACLES)]
    obstacles_y = [random.randrange(settings.MIN_OBSTACLE_Y_POSITION,
                                    (settings.MAP_SIZE - 1) - 5,
                                    settings.MIN_DISTANCE_BETWEEN_OBSTACLES) for _ in range(settings.MAX_NUM_OBSTACLES)]

    obstacles = list(zip(obstacles_x, obstacles_y))

    # Remove any obstacles that are too close to the robot's starting position
    relative_positions = obstacles - np.tile(x_start, (settings.MAX_NUM_OBSTACLES, 1))
    distances = np.sqrt(np.sum(np.square(relative_positions), axis=1))
    obstacles = np.delete(obstacles, np.argwhere(distances < settings.EXCLUSION_ZONE_RADIUS), axis=0)
    # Remove obstacles directly ahead of starting position
    obstacles = np.delete(obstacles, np.argwhere(obstacles[:, 0] == x_start[0]), axis=0)

    # CUSTOM OBSTACLES FOR DEBUGGING ONLY
    # obstacles = np.append(obstacles, np.array([[17, 17], [18, 17], [19, 17], [20, 17]]), axis=0)

    num_obstacles = len(obstacles)  # account for the loss of the obstacles that were too close
    # Export obstacle location to disk
    np.savetxt(("%s%s%s%s%s%s" % (split_data_path, "/obstacles_", data_subset_type, "_", idx, ".csv")),
               obstacles, delimiter=",")

    # The scene context will affect robot behaviour, ranging from unimpeded driving to being at a standstill
    scene_context_constant = np.random.uniform(0.0001, 1.0)
    while k < settings.MAX_ITERATIONS:
        relative_positions = obstacles - np.tile(x_robot, (num_obstacles, 1))

        for i in range(len(relative_positions)):
            if np.all(relative_positions[i, :] == [0, 0]):
                print("Error: obstacle and robot are both at position:", obstacles[i, :],
                      "so will now move robot slightly away from this")
                relative_positions[i, :] = [1e-3, 1e-3]
                print("k = ", k)
        distances = np.sqrt(np.sum(np.square(relative_positions), axis=1))
        idx_proximal = distances < settings.OBSTACLE_INFLUENCE_RADIUS

        if any(idx_proximal):
            rho = np.tile(distances[idx_proximal].reshape(1, -1).transpose(), (1, 2))
            v = relative_positions[idx_proximal, :]
            d_rho_dx = -v / rho
            f_proximity = (1 / rho - 1 / settings.OBSTACLE_INFLUENCE_RADIUS) * 1 / (np.square(rho)) * d_rho_dx

            # Force due to obstacle "visibility" (high if in front of robot, low if to the side or behind)
            robot_angle_from_vertical = (robot_th[-1] - robot_th[0])
            relative_angles = -np.arctan2(relative_positions[idx_proximal, 0],
                                          relative_positions[idx_proximal, 1]) - robot_angle_from_vertical
            relative_angles[relative_angles > np.pi] = relative_angles[relative_angles > np.pi] - 2 * np.pi
            relative_angles[relative_angles < -np.pi] = 2 * np.pi - relative_angles[relative_angles < -np.pi]
            angle_influence = (np.pi - np.abs(relative_angles)) / np.pi

            # Force due to relevance in the goal quest - obstacles that are not still ahead are given less importance
            relevance_to_mission = relative_positions[idx_proximal, 1]
            relevance_to_mission[relevance_to_mission < 0] = 0
            relevance_to_mission[relevance_to_mission > 0] = 1

            f_proximity *= (1 / 3 + angle_influence / 3 + relevance_to_mission / 3).reshape(-1, 1)
            f_objects = settings.OBSTACLE_FORCE_MULTIPLIER * np.sum(f_proximity, axis=0)  # .reshape(-1, 1)
        else:
            f_objects = np.array([0, 0])
        f_goal = settings.GOAL_FORCE_MULTIPLIER * goal_error / np.linalg.norm(goal_error)
        f_total = f_goal + f_objects

        # This takes into account the natural slow-down that should happen when the robot is near obstacles (caution)
        max_velocity = min(settings.VELOCITY_LIMIT,
                           settings.NOMINAL_VELOCITY * min(distances) / settings.PROXIMITY_TO_OBSTACLE_CAUTION_FACTOR)
        f_total = f_total / np.linalg.norm(f_total) * min(max_velocity, np.linalg.norm(f_total))
        f_total *= scene_context_constant
        theta_robot = math.atan2(f_total[1], f_total[0])
        x_robot = x_robot + f_total

        robot_xy = np.append(robot_xy, x_robot.reshape(1, -1), axis=0)
        robot_th = np.append(robot_th, theta_robot)
        goal_error = x_goal - x_robot
        k += 1

    if save_plots:
        plt.figure(figsize=(10, 10))
        colours = ["red", "blue", "magenta", "green"]
        plt.plot(obstacles[:, 0], obstacles[:, 1], "*", color=colours[0], label="obstacles")
        draw_robot_poses(robot_xy[:, 0], robot_xy[:, 1], robot_th, colours[1])
        plt.plot(x_start[0], x_start[1], "o", color=colours[2])
        plt.plot(x_goal[0], x_goal[1], "o", color=colours[3])
        plt.grid()
        plt.xlim(0, settings.MAP_SIZE)
        plt.ylim(0, settings.MAP_SIZE)
        lines = [plt.Line2D([0], [0], color=c, linewidth=0, marker='.', markersize=20) for c in colours]
        labels = ["Obstacles", "True pose", "Start", "Goal"]
        plt.legend(lines, labels)
        plt.savefig("%s%s%s%s%i%s" % (split_data_path, "/", data_subset_type, "_maze_", idx, ".pdf"))
        plt.close()
    if idx % 100 == 0:
        print("Maze sim complete for", data_subset_type, "index:", idx)

    return robot_xy, robot_th, obstacles


def draw_robot_poses(x_poses, y_poses, thetas, colour):
    scale = 0.25
    # basic_triangle = np.array([[-1, -1], [0, 2], [1, -1]]) * scale
    basic_triangle = np.array([[-1, -1], [2, 0], [-1, 1]]) * scale
    # print(basic_triangle)

    for i in range(len(x_poses)):
        th = -thetas[i]  # -(thetas[i] - np.pi / 2)
        rotation_matrix = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        triangle_vertices = np.matmul(basic_triangle, rotation_matrix)
        triangle_vertices[:, 0] = triangle_vertices[:, 0] + x_poses[i]
        triangle_vertices[:, 1] = triangle_vertices[:, 1] + y_poses[i]
        triangle = plt.Polygon(triangle_vertices, color=colour, alpha=1.0, fill=False)
        plt.gca().plot(x_poses[i], y_poses[i], '.', color=colour, alpha=1.0)
        plt.gca().add_patch(triangle)


def generate_relative_poses(robot_xy, robot_th):
    robot_global_poses = []
    relative_poses = []

    for i in range(len(robot_xy)):
        T_i = np.identity(4)
        th = robot_th[i]
        T_i[0, 0] = np.cos(th)
        T_i[0, 1] = -np.sin(th)
        T_i[1, 0] = np.sin(th)
        T_i[1, 1] = np.cos(th)
        T_i[0, 3] = robot_xy[i, :][0]
        T_i[1, 3] = robot_xy[i, :][1]
        robot_global_poses.append(T_i)

    # Loop runs for 1 less iteration because these are relative poses
    for i in range(1, len(robot_global_poses)):
        T_rel_pose = np.linalg.inv(robot_global_poses[i - 1]) @ robot_global_poses[i]
        # Manually adjust transform to be in robot frame
        T_i = np.identity(4)
        th = -np.arctan2(T_rel_pose[1, 0], T_rel_pose[1, 1])
        T_i[0, 0] = np.cos(th)
        T_i[0, 1] = -np.sin(th)
        T_i[1, 0] = np.sin(th)
        T_i[1, 1] = np.cos(th)
        T_i[0, 3] = T_rel_pose[0, 3]
        T_i[1, 3] = -T_rel_pose[1, 3]
        relative_poses.append(T_i)

    return relative_poses


def generate_maze_samples(num_samples, data_subset_type):
    split_data_path = Path(settings.MAZE_IMAGE_DIR, data_subset_type)
    if split_data_path.exists() and split_data_path.is_dir():
        shutil.rmtree(split_data_path)
    split_data_path.mkdir(parents=True)
    save_plots = False
    print_timing_info = False

    for idx in range(num_samples):
        t0_start = time.time()
        xy_positions, thetas, t0_obstacles = run_maze_sim_and_generate_images(idx, split_data_path, data_subset_type,
                                                                              save_plots)
        t0_end = time.time() - t0_start
        t1_start = time.time()
        relative_poses = generate_relative_poses(xy_positions, thetas)
        t1_end = time.time() - t1_start
        t2_start = time.time()
        tp1_obstacles, relative_poses = generate_future_obstacle_positions(relative_poses, idx, split_data_path,
                                                                           data_subset_type,
                                                                           save_obstacle_plots=save_plots)
        t2_end = time.time() - t2_start
        t3_start = time.time()
        save_relative_poses_as_labels(relative_poses, split_data_path, data_subset_type, idx, save_plots=save_plots)
        t3_end = time.time() - t3_start
        t4_start = time.time()
        save_obstacles_as_images(t0_obstacles, split_data_path, data_subset_type, idx, time_frame="t0")
        save_obstacles_as_images(tp1_obstacles, split_data_path, data_subset_type, idx, time_frame="tp1")
        t4_end = time.time() - t4_start
        total_time = time.time() - t0_start
        if print_timing_info:
            print("run_maze_sim_and_generate_images duration:", t0_end)
            print("generate_relative_poses duration:", t1_end)
            print("generate_future_obstacle_positions duration:", t2_end)
            print("save_relative_poses_as_labels duration:", t3_end)
            print("save_obstacles_as_images duration:", t4_end)
            print("Total time: ", total_time)

    print("Generated", num_samples, data_subset_type, "samples, with dim =", settings.MAZE_IMAGE_DIMENSION,
          "and written to:", split_data_path)


def save_relative_poses_as_labels(relative_poses, split_data_path, data_subset_type, idx, save_plots=False):
    csv_file = "%s%s%s%s%s%s" % (split_data_path, "/speed_labels_", data_subset_type, "_", idx, ".csv")
    with open(csv_file, 'w') as pose_labels_file:
        wr = csv.writer(pose_labels_file, delimiter=",")
        for i in range(len(relative_poses)):
            pose_label = [relative_poses[i][0, 3], relative_poses[i][1, 3],
                          np.arctan2(relative_poses[i][1, 0], relative_poses[i][1, 1])]
            wr.writerow(pose_label)

    if save_plots:
        # Open the file we've just written to, and plot the values
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            pose_data = list(reader)

        dx = [float(items[0]) for items in pose_data]
        dy = [float(items[1]) for items in pose_data]
        dth = [float(items[2]) for items in pose_data]

        plt.figure(figsize=(10, 3))
        plt.plot(dx, '.-', label="x")
        plt.plot(dy, '.-', label="y")
        plt.plot(dth, '.-', label="yaw")
        plt.grid()
        plt.legend()
        plt.savefig("%s%s%i%s" % (split_data_path, "/dx_dy_dth_", idx, ".png"))
        plt.close()


def generate_future_obstacle_positions(relative_poses, idx, split_data_path, data_subset_type,
                                       save_obstacle_plots=False):
    T_origin = np.identity(4)
    th_origin = 0
    T_origin[0, 0] = np.cos(th_origin)
    T_origin[0, 1] = -np.sin(th_origin)
    T_origin[1, 0] = np.sin(th_origin)
    T_origin[1, 1] = np.cos(th_origin)
    T_origin[0, 3] = settings.MAP_SIZE / 2
    T_origin[1, 3] = settings.MAP_SIZE / 2

    # # Manual transform for debugging...
    # T_i = np.identity(4)
    # th = np.pi/32
    # x = 0
    # y = 0
    # T_i[0, 0] = np.cos(th)
    # T_i[0, 1] = -np.sin(th)
    # T_i[1, 0] = np.sin(th)
    # T_i[1, 1] = np.cos(th)
    # T_i[0, 3] = -y
    # T_i[1, 3] = -x

    # Manually adjust transform to be in global frame for moving obstacles (checked with debugging matrix above)
    T_i = np.identity(4)
    th = np.arctan2(relative_poses[0][1, 0], relative_poses[0][1, 1])
    x = relative_poses[0][0, 3]
    y = relative_poses[0][1, 3]
    T_i[0, 0] = np.cos(th)
    T_i[0, 1] = -np.sin(th)
    T_i[1, 0] = np.sin(th)
    T_i[1, 1] = np.cos(th)
    T_i[0, 3] = -y
    T_i[1, 3] = -x

    first_relative_pose = T_i
    obstacles = np.genfromtxt(
        settings.MAZE_IMAGE_DIR + data_subset_type + "/obstacles_" + data_subset_type + "_" + str(idx) + ".csv",
        delimiter=",")
    obs_arr = np.array(obstacles)
    obs_arr = np.c_[obs_arr, np.zeros([len(obs_arr), 1]), np.ones([len(obs_arr), 1])]
    tp1_obstacles = []
    for i in range(len(obstacles)):
        new_obstacle_position = T_origin @ first_relative_pose @ np.linalg.inv(T_origin) @ obs_arr[i, :]
        tp1_obstacles.append(new_obstacle_position)
    # Bump off 1 relative pose that was used to generate the previous motion so that the labels only carry future poses
    # and not the previous one(s)
    relative_poses = relative_poses[1:]
    tp1_obstacles = np.array(tp1_obstacles)[:, 0:2]
    np.savetxt(("%s%s%s%s%s%s" % (split_data_path, "/tp1_obstacles_", data_subset_type, "_", idx, ".csv")),
               tp1_obstacles, delimiter=",")

    if save_obstacle_plots:
        plt.figure(figsize=(10, 10))
        colours = ["red", "blue"]
        plt.plot(obstacles[:, 0], obstacles[:, 1], "*", color=colours[0], label="t0 obstacles")
        plt.plot(tp1_obstacles[:, 0], tp1_obstacles[:, 1], ".", color=colours[1], label="tp1 obstacles")
        plt.grid()
        plt.xlim(0, settings.MAP_SIZE)
        plt.ylim(0, settings.MAP_SIZE)
        lines = [plt.Line2D([0], [0], color=c, linewidth=0, marker='.', markersize=20) for c in colours]
        labels = ["t0 obstacles", "tp1 obstacles"]
        plt.legend(lines, labels)
        plt.savefig("%s%s%s%s%i%s" % (split_data_path, "/", data_subset_type, "_obstacles_", idx, ".pdf"))
        plt.close()
        print("Saved obstacle plot for index:", idx)

    return tp1_obstacles, relative_poses


def save_obstacles_as_images(obstacles, split_data_path, data_subset_type, idx, time_frame):
    if time_frame is "tp1":
        obstacles = np.round(obstacles).astype(int)
    data = np.zeros((settings.MAP_SIZE, settings.MAP_SIZE), dtype=np.uint8)
    radius = settings.ADDITIONAL_OBSTACLE_VISUAL_WEIGHT
    for i in range(len(obstacles)):
        data[(settings.MAP_SIZE - 1) - obstacles[i, 1] - radius:
             (settings.MAP_SIZE - 1) - obstacles[i, 1] + radius + 1,
        obstacles[i, 0] - radius:obstacles[i, 0] + radius + 1] = 255
    img = Image.fromarray(data, 'L')
    img.save("%s%s%s%s%i%s%s%s" % (split_data_path, "/", data_subset_type, "_", idx, "_", time_frame, ".png"))
    img.close()


if __name__ == "__main__":
    start_time = time.time()
    # import pdb
    # pdb.set_trace()
    generate_maze_samples(settings.TRAIN_SET_SIZE, settings.TRAIN_SUBSET)
    generate_maze_samples(settings.VAL_SET_SIZE, settings.VAL_SUBSET)
    generate_maze_samples(settings.TEST_SET_SIZE, settings.TEST_SUBSET)
    print("--- Dataset generation execution time: %s seconds ---" % (time.time() - start_time))
