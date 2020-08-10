import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import settings

np.random.seed(0)
obstacles = np.random.randint(1, settings.MAP_SIZE, size=(2, settings.NUM_OBSTACLES))

subset_fig_path = settings.MAZE_IMAGE_DIR + "tmp" + "/"
Path(subset_fig_path).mkdir(parents=True, exist_ok=True)


def run_maze_sim():
    k = 0
    x_start = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE / 2]).reshape(2, -1)
    x_goal = np.array([settings.MAP_SIZE / 2, settings.MAP_SIZE]).reshape(2, -1)
    x_robot = x_start
    goal_error = x_goal - x_robot
    robot_xy = np.array(x_robot)
    robot_th = np.array(0)
    plt.figure(figsize=(10, 10))

    while np.linalg.norm(goal_error) > 1 and k < settings.MAX_ITERATIONS:
        relative_positions = obstacles - np.tile(x_robot, (1, settings.NUM_OBSTACLES))
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
            f_objects = settings.OBSTACLE_WEIGHT * np.sum(f_proximity, axis=1).reshape(-1, 1)
        else:
            f_objects = np.array([0, 0]).reshape(-1, 1)
        f_goal = settings.GOAL_WEIGHT * goal_error / np.linalg.norm(goal_error)
        f_total = f_goal + f_objects
        f_total = f_total / np.linalg.norm(f_total) * min(settings.VELOCITY_LIMIT, np.linalg.norm(f_total))
        theta_robot = math.atan2(f_total[1], f_total[0])
        x_robot = x_robot + f_total
        robot_xy = np.append(robot_xy, x_robot, axis=1)
        robot_th = np.append(robot_th, theta_robot)
        plt.plot(x_robot[0], x_robot[1], 'b^')
        goal_error = x_goal - x_robot
        k += 1

    plt.plot(obstacles[0, :], obstacles[1, :], 'r*')
    plt.plot(x_goal[0], x_goal[1], 'go')
    plt.grid()
    plt.savefig("%s%s%i%s" % (subset_fig_path, "idx-", 0, ".png"))
    plt.close()
    print("k =", k)
    return robot_xy, robot_th


def generate_image_from_maze():
    data = np.zeros((settings.MAP_SIZE, settings.MAP_SIZE), dtype=np.uint8)
    # obstacle_radius = 1
    for i in range(settings.NUM_OBSTACLES):
        # data[
        # settings.MAP_SIZE - obstacles[1, i] - obstacle_radius:settings.MAP_SIZE - obstacles[1, i] + obstacle_radius,
        # obstacles[0, i] - obstacle_radius:obstacles[0, i] + obstacle_radius] = 255
        data[settings.MAP_SIZE - obstacles[1, i] - 1,
             obstacles[0, i]] = 255
    img = Image.fromarray(data, 'L')
    img.save("%s%s%i%s" % (subset_fig_path, "image-", 0, ".png"))


def generate_relative_poses(robot_xy, robot_th):
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
    for i in range(len(relative_poses)):
        dx.append(relative_poses[i][0, 2])
        dy.append(relative_poses[i][1, 2])
    plt.figure(figsize=(10, 3))
    plt.plot(dx, '.-')
    plt.plot(dy, '.-')
    plt.grid()
    plt.savefig("%s%s%i%s" % (subset_fig_path, "dx_dy-", 0, ".png"))


xy_positions, thetas = run_maze_sim()
generate_image_from_maze()
generate_relative_poses(xy_positions, thetas)
