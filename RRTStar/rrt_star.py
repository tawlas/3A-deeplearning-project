"""
Path planning Sample Code with RRT*
author: Atsushi Sakai(@Atsushi_twi)
"""

import math
import os
import sys
import json
from tqdm import tqdm

import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor as Pool
import concurrent.futures

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

show_animation = True


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, rand_area, obstacle_radius,
                 expand_dis=5.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=250,
                 connect_circle_dist=50.0
                 ):
        super().__init__(start, goal, obstacle_list,
                         rand_area, obstacle_radius, expand_dis, path_resolution, goal_sample_rate, max_iter)
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])

    def planning(self, animation=False, search_until_max_iter=False):
        """
        rrt star path planning
        animation: flag for animation on or off
        search_until_max_iter: search until max iteration for path improving or not
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            # print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(
                self.node_list[nearest_ind], rnd, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            if animation and i % 5 == 0:
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    # print("Found after iteration: ", i)
                    return self.generate_final_course(last_index)

        # print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(
            n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(
            i) for i in dist_to_goal_list if i <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node = edge_node
                near_node.parent = new_node
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main(sg_path, coords_path, output_dir):
    print("Start " + __file__)
    coords_all = json.load(open(coords_path, "r"))
    sg_all = json.load(open(sg_path, "r"))
    img_name_list = sorted(coords_all.keys())
    n_attempt = 1  # number of attempt to find a path in case path is not found at the first iteration

    for img_name in tqdm(img_name_list):
        trajectories = {}
        coords_list = coords_all[img_name]
        sg_list = sg_all[img_name]
        for idx, sg in enumerate(tqdm(sg_list)):
            start = sg[0]
            goal = sg[1]

            # ====Search Path with RRT====

            # Set Initial parameters
            path = None
            i = 0
            while i < n_attempt and path is None:
                rrt_star = RRTStar(start=start,
                                   goal=goal,
                                   rand_area=[0, 63],
                                   obstacle_list=coords_list,
                                   obstacle_radius=0.6)
                path = rrt_star.planning(animation=show_animation)
                i += 1

            if path is None:
                print("Cannot find path for {} with the following set of start and goal: {}".format(
                    img_name, sg))
            else:
                trajectories[idx] = path
                # print("found path!!")
                # Draw final path
                if show_animation:
                    rrt_star.draw_graph()
                    plt.plot([x for (x, y) in path], [
                             y for (x, y) in path], '-r')
                    plt.grid(True)
                    plt.pause(0.01)  # Need for Mac
                    plt.show()

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        traj_file = os.path.join(output_dir, img_name[:-4]+".json")
        json.dump(trajectories, open(traj_file, "w"))


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description="Start and Goal")
    arg_parser.add_argument(
        "--sg_path",
        "-sg",
        dest="sg_path",
        required=True
    )
    arg_parser.add_argument(
        "--coords_path",
        "-c",
        dest="coords_path",
        required=True
    )
    arg_parser.add_argument(
        "--output_dir",
        "-o",
        dest="output_dir",
        required=True
    )

    args = arg_parser.parse_args()
    main(args.sg_path, args.output_dir)
