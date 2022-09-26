#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
"""
import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node as RclpyNode
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid

# TODO: import as you need
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
import random
from ament_index_python.packages import get_package_share_directory
import pathlib
from scipy.interpolate import splprep, splev
from visualization_msgs.msg import MarkerArray, Marker
import cv2
from cv_bridge import CvBridge, CvBridgeError
from skimage import img_as_ubyte
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
# from skimage import img_as_ubyte
# class def for tree nodes
# It's up to you if you want to use this
class Node(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.parent = None
        self.cost = None # only used in RRT*
        self.is_root = False

# class def for RRT
class RRT(RclpyNode):
    def __init__(self):
        super().__init__('rrt_node')
        # topics, not saved as attributes
        # TODO: grab topics from param file, you'll need to change the yaml file
        pose_topic = "/ego_racecar/odom"
        # pose_topic = "/odom"
        scan_topic = "/scan"
        occ_topic = "/rrt/occ_grid"
        drive_topic = "/drive"
        waypoint_topic = '/pure_pursuit/waypoint'
        rrt_goal_waypoint_topic = '/rrt/rrt_goal_waypoint'
        waypointmap_topic = '/pure_pursuit/waypoint_map'
        rrt_node_topic = '/rrt/nodes'
        # you could add your own parameters to the rrt_params.yaml file,
        # and get them here as class attributes as shown above.

        # create subscribers
        self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
        self.scan_sub_ = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 1)

        # publishers
        self.AckPublisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.OccMapPublisher = self.create_publisher(OccupancyGrid, occ_topic, 10)
        self.WaypointVisualizer = self.create_publisher(Marker, waypoint_topic, 10)
        self.RrtWaypointVisualizer = self.create_publisher(Marker, rrt_goal_waypoint_topic, 10)
        self.WaypointMapvisualizer = self.create_publisher(MarkerArray, waypointmap_topic, 10)
        self.RrtPathVisualizer = self.create_publisher(MarkerArray, rrt_node_topic, 10)

        #parameters
        self.declare_parameter('rrt_lookahead_distance')
        self.declare_parameter('pursuit_lookahead_distance')
        self.declare_parameter('max_iter')
        self.declare_parameter('max_steer_dist')
        self.declare_parameter('goal_dist_thresh')
        self.declare_parameter('grid_resolution')
        self.declare_parameter('grid_width')
        self.declare_parameter('grid_length')
        self.declare_parameter('waypoint_distance')
        self.declare_parameter('desired_speed')
        self.declare_parameter('min_speed')
        self.declare_parameter('steering_angle_factor')
        self.declare_parameter('speed_factor')
        self.declare_parameter('sparse_filename')

        # class attributes
        # TODO: maybe create your occupancy grid here
        self.grid_res = self.get_parameter('grid_resolution').value
        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_length').value
        self.y_grid = int(np.ceil(self.grid_width / self.grid_res))
        self.x_grid = int(np.ceil(self.grid_height / self.grid_res))
        # self.grid_size = (self.x_grid, self.y_grid)
        self.half_grid = self.x_grid // 2
        self.rear_lidar = 0.29275
        #create occupancy map
        # self.occupancy_grid = np.ones((int(self.x_grid), int(self.y_grid)))

        #inital position and orientation
        self.x_cur = 0
        self.y_cur = 0
        self.yaw = 0

        #Get waypoints and make path
        pkg_dir = get_package_share_directory('lab7_pkg')
        filepath = pkg_dir + '/waypoints/' + self.get_parameter('sparse_filename').value + '.csv'
        if not pathlib.Path(filepath).is_file():
            pathlib.Path(filepath).touch()
        data = np.genfromtxt(filepath, delimiter=',')
        self.path = self.generate_waypoint_path(data, self.get_parameter('waypoint_distance').value)
        self.timer = self.create_timer(1.0, self.publish_dense_map)
        # self.timer = self.create_timer(1.0, self.publish_waypoint_map_msg)
        # self.image_pub = rospy.Publisher("/occ_grid", Image, queue_size=10)

        #new stuff
        angle_min = -3.141592741
        angle_inc = 0.005823155
        n = 1080
        self.world_size = (500, 200)
        self.occupancy_grid = np.ones(self.world_size)

        self.angles = np.array([angle_min + i * angle_inc for i in range(n)])
        self.plot_width = 12
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image,"/occ_grid",10)

    
    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:
        """

        # print("In scan_callback.")
        # lidar_x = self.x_cur + self.rear_lidar * np.cos(self.yaw)
        # lidar_y = self.y_cur + self.rear_lidar * np.sin(self.yaw)
        # # lidar_x = 0
        # # lidar_y = 0

        # curr_angle = self.yaw + scan_msg.angle_min
        # self.occupancy_grid = np.ones((int(self.x_grid), int(self.y_grid)))
        # for i in range(len(scan_msg.ranges)):
        #     curr_angle += scan_msg.angle_increment
        #     if curr_angle < -np.pi/2 or curr_angle > np.pi/2:
        #         continue
            
        #     x_obs = (lidar_x + (scan_msg.ranges[i] * np.cos(curr_angle)) / self.get_parameter('grid_resolution').value) 
        #     y_obs = (lidar_y + (scan_msg.ranges[i] * np.sin(curr_angle)) / self.get_parameter('grid_resolution').value) + self.half_grid
        #     self.occupancy_grid[int(np.floor(x_obs)):int(np.ceil(x_obs)) + 1, int(np.floor(y_obs)):int(np.ceil(y_obs)) + 1] = 0
        # print("Occupangy grid built.")
        # self.publish_occ_grid(self.OccMapPublisher, self.occupancy_grid, self.grid_res, [self.x_grid, self.y_grid], 
        #         origin=[self.rear_lidar, -self.grid_width / 2, 0.0], frame='ego_racecar/base_link')

        #NEW VERSION
        print("In scan_callback.")
        dist =scan_msg.ranges[179:899]
        # self.occupancy_grid = np.ones(self.world_size)
        working_occ_grid = np.ones(self.world_size)
        lidar_x = self.x_cur + self.rear_lidar * np.cos(self.yaw)
        lidar_y = self.y_cur + self.rear_lidar * np.sin(self.yaw)

        curr_angle = self.yaw + self.angles[179:899]
        x_obs =lidar_x + dist * np.cos(curr_angle)
        y_obs = lidar_y + dist * np.sin(curr_angle)

        x_off = 14.5
        y_off = 0.7
        # x_off = 0
        # y_off = 0

        x_grid =(x_obs+x_off) / self.grid_res
        y_grid = (y_obs+y_off) / self.grid_res

        x_grid[(x_grid >= 499.5)] = 499
        y_grid[(y_grid >= 199.5)] = 199
        x_grid[((x_grid < 0))] = 0
        y_grid[((y_grid < 0))] = 0

        x_grid =x_grid.round(0).astype(int)
        y_grid = y_grid.round(0).astype(int)

        for i in range(len(dist)):
            x,y = x_grid[i],y_grid[i]
            working_occ_grid[max(x-self.plot_width//2,0):min(x+self.plot_width//2,self.world_size[0]-1)+1,max(y-self.plot_width//2,0):min(y+self.plot_width//2,self.world_size[1]-1)+1] = 0
            reverse_grid = np.flipud(np.fliplr(working_occ_grid))
            _, grid_img = cv2.threshold(reverse_grid, 0, 1, cv2.THRESH_BINARY)
            cv_image = img_as_ubyte(grid_img)
        self.occupancy_grid = working_occ_grid
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'mono8'))
        except CvBridgeError as e:
            print(e)
        print("Occupangy grid built.")
        
        return 0

    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """
        print("pose_callback")
        #get yaw
        pose_orientation = pose_msg.pose.pose.orientation
        quarternion = [pose_orientation.x, pose_orientation.y, pose_orientation.z, pose_orientation.w]
        roll, pitch, yaw = euler_from_quaternion(quarternion)
        self.yaw = yaw

        self.x_cur = pose_msg.pose.pose.position.x
        self.y_cur = pose_msg.pose.pose.position.y
        #initalize tree
        start = Node()
        start.x = self.x_cur
        start.y = self.y_cur
        start.parent = -1
        start.loss = None
        start.is_root = True
        tree = [start]
        
        count = 0
        for _ in range(self.get_parameter('max_iter').value):
            #sample point
            sampled_pt = self.sample(pose_msg)
            # print('sampled point', sampled_pt)
            #find nearest point of sample point in tree
            nearest_node_ind = self.nearest(tree, sampled_pt)
            # print('tree', tree)
            # print('nearest node index', nearest_node_ind)
            #expand the tree towards the sample point created
            new_node = self.steer(tree[nearest_node_ind], sampled_pt)
            new_node.parent = nearest_node_ind
            # print('new node x:', new_node.x)
            #check whether collision will occur between tree point and new point
            test1 = Node()
            test2 = Node()
            test1.x = 1.0
            test1.y = 4.0
            test2.x = 0.0
            test2.y = 0.0
            # self.publish_waypoint_msg(self.RrtWaypointVisualizer, [test1.x, test1.y], [255.0, 0.0, 0.0, 1.0])
            # self.publish_waypoint_msg(self.RrtWaypointVisualizer, [test2.x, test2.y], [0.0, 255.0, 0.0, 1.0])
            # if(self.check_edge_collision(test1, test2)):
                # self.get_logger().warn("collision occurs")
            if(self.check_edge_collision(tree[nearest_node_ind], new_node)):
                self.get_logger().warn("collision occurs")
                continue
            self.get_logger().warn("no collision")
            #append node into tree
            tree.append(new_node)
            #Determines rrt goal point
            rrt_x_goal, rrt_y_goal = self.find_rrt_goal(pose_msg, self.path, self.get_parameter('rrt_lookahead_distance').value) 
            #Visualizing the RRT goal 
            self.publish_waypoint_msg(self.RrtWaypointVisualizer, [rrt_x_goal, rrt_y_goal], [255.0, 0.0, 0.0, 1.0])
            # print(f'rrt x goal = {rrt_x_goal}, rrt y goal = {rrt_y_goal}')
            #check if it is possible to reach goal from current point
            tree_nodes = np.array([[x.x, x.y, 1] for x in tree])
            #publishing node 
            # self.publish_waypoint_map_msg(self.RrtPathVisualizer, tree_nodes, [0.0, 0.0, 255.0, 1.0])
            # msg = AckermannDriveStamped()
            # msg.drive.steering_angle = 0.0
            # msg.drive.speed = 0.0
            # self.AckPublisher.publish(msg)
            if (self.is_goal(new_node, rrt_x_goal, rrt_y_goal)):
                self.rrt_path = self.find_path(tree, new_node)
                path_nodes = np.array([[x.x, x.y, 1] for x in self.rrt_path])
                #publishing node markers
                self.publish_waypoint_map_msg(self.RrtPathVisualizer, path_nodes, [255.0, 255.0, 255.0, 1.0])
                #transforming from map to car:
                path_nodes_car = []
                for node in self.rrt_path:
                    x, y = self.transform_point(pose_msg, node.x, node.y)
                    path_nodes_car.append([x,y])
                # print(path_nodes)
                path_norms = np.array([np.linalg.norm([x[0], x[1]]) for x in path_nodes_car])
                # Use pure pursuit lookahead to see where path intersects with rrt path
                # TODO: CHANGE method to interpolate between path values
                # self.goal_idx = np.argmin(np.abs(self.get_parameter('pursuit_lookahead_distance').value - path_norms[1:]))
                # x_goal = self.rrt_path[self.goal_idx + 1].x
                # y_goal = self.rrt_path[self.goal_idx + 1].y
                x_goal, y_goal = self.find_rrt_goal(pose_msg, path_nodes, self.get_parameter('pursuit_lookahead_distance').value) 
                self.publish_waypoint_msg(self.WaypointVisualizer, [x_goal, y_goal], [0.0, 255.0, 0.0, 1.0])
                # print(f'x goal = {x_goal}, y goal = {y_goal}')
                #TODO: Convert global waypoint to car frame
                goal_x_body, goal_y_body = self.transform_point(pose_msg, x_goal, y_goal)
                #publish steering
                steering_angle = 2 * goal_y_body / self.get_parameter('pursuit_lookahead_distance').value**2  
                self.publish_drive_msg(steering_angle)
                ##Visualizing the pure pursuit goal 
            # else:
                #
                # msg = AckermannDriveStamped()
                # msg.drive.steering_angle = 0.0
                # msg.drive.speed = 0.0
                # self.AckPublisher.publish(msg)
                return 0
        
        self.get_logger().warn("Max iterations reached!")
        return 1

    def sample(self, odom_msg):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (float float): a tuple representing the sampled point

        """
        #range in front of car
        car_x = random.uniform(0, self.get_parameter('rrt_lookahead_distance').value)
        #range from left to right side of bubble
        car_y = random.uniform(-self.get_parameter('rrt_lookahead_distance').value, self.get_parameter('rrt_lookahead_distance').value)
        x, y = self.transform_car_to_global(odom_msg, car_x, car_y)
        return (x, y)

    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node (int): index of neareset node on the tree
        """
        min_dist = float('inf')
        for i in range(len(tree)):
            dist = np.linalg.norm([sampled_point[0] - tree[i].x, sampled_point[1] - tree[i].y])
            if dist < min_dist:
                min_dist = dist
                nearest_node = i
        return nearest_node

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (Node): nearest node on the tree to the sampled point
            sampled_point (tuple of (float, float)): sampled point
        Returns:
            new_node (Node): new node created from steering
        """
        
        x_dis = sampled_point[0] - nearest_node.x
        y_dis = sampled_point[1] - nearest_node.y
        dist = np.linalg.norm([x_dis, y_dis])
        new_node = Node()
        new_node.x = nearest_node.x + min(dist, self.get_parameter('max_steer_dist').value) * (x_dis / dist)
        new_node.y = nearest_node.y + min(dist, self.get_parameter('max_steer_dist').value) * (y_dis / dist)
        return new_node

    def check_edge_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest (Node): nearest node on the tree
            new_node (Node): new node from steering
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        start = np.array([nearest_node.x, nearest_node.y])
        goal = np.array([new_node.x, new_node.y])
        d = np.linalg.norm(goal - start)
        n_points = int(np.ceil(d) / self.grid_res)
        if n_points > 1:
            step = d / (n_points - 1)
            v = goal - start
            u = v / np.linalg.norm(v)
            for i in range(n_points):
                if (self.check_collision(start[0], start[1])):
                    return True
                start += u * step
        return False

        # increment = np.arange(0, 1.01, 0.01)
        # x = nearest_node.x + increment * 0.01 * (new_node.x - nearest_node.x)
        # y = nearest_node.y + increment * 0.01 * (new_node.y - nearest_node.y)
        # # x = new_node.x - 
        # grid_x, grid_y = self.global_to_grid(x, y)
        # grid_x, grid_y = np.unique(grid_x), np.unique(grid_y)
        # for x in grid_x:
        #     for y in grid_y:
        #         if self.occupancy_grid[x][y] == 0:
        #             return True
        #             break
        # return False

        # x_diff = int((new_node.x - nearest_node.x) / self.get_parameter('grid_resolution').value)
        # y_diff = int((new_node.y - nearest_node.y) / self.get_parameter('grid_resolution').value)
        # if x_diff == 0:
        #     x_diff = 1
        # slope_low = int(np.floor(y_diff / x_diff))
        # slope_high = int(np.ceil(y_diff / x_diff))
        # cur_x = int(nearest_node.x)
        # cur_y_low = int(nearest_node.y)
        # cur_y_high = int(nearest_node.y)
        # for x in range(x_diff + 1):
        #     if (self.occupancy_grid[cur_x, cur_y_low] == 0) or (self.occupancy_grid[cur_x, cur_y_high] == 0):
        #         return True
        #         break 
        #     cur_x += x
        #     cur_y_low += slope_low
        #     cur_y_high += slope_high
        # return False

    def check_collision(self, x, y):
        """
        """
        r, c = self.global_to_grid(x, y)
        return self.occupancy_grid[r, c] == 0

    def global_to_grid(self, x_global, y_global):
        x_off = 14.5
        y_off = 0.7
        x_grid = (x_global + x_off) / self.grid_res
        y_grid = (y_global + y_off) / self.grid_res
        if (x_grid >= 499.5):
            x_grid = 499
        elif (x_grid < 0):
            x_grid = 0
        if (y_grid >= 199.5):
            y_grid = 199
        elif (y_grid < 0):
            y_grid = 0
        return np.round(x_grid).astype(int), np.round(y_grid).astype(int)

    def is_goal(self, latest_added_node, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            latest_added_node (Node): latest added node on the tree
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enoughg to the goal
        """
        dist = np.linalg.norm([goal_x - latest_added_node.x, goal_y - latest_added_node.y])
        return dist < self.get_parameter('goal_dist_thresh').value

    def find_path(self, tree, latest_added_node):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node (Node): latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = []
        path.append(latest_added_node)
        next_node = tree[latest_added_node.parent]
        while (not next_node.is_root):
            path.append(next_node)
            # path.append(next_node.parent)
            next_node = tree[next_node.parent]
        path.append(tree[0])
        path.reverse()
        return path
    
    def find_rrt_goal(self, odom_msg, path, lookahead):
        """
        Find waypoint to track on based on lookahead distance.
        """
        # print(path)
        position = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        point_dist = np.linalg.norm(path[:, 0:2] - position, axis=1)

        search_index = np.argmin(point_dist)
        while True:
            search_index = (search_index + 1) % np.size(path, axis=0)
            dis_diff = point_dist[search_index] - lookahead
            if dis_diff < 0:
                continue
            elif dis_diff > 0:
                x_goal = np.interp(self.get_parameter('rrt_lookahead_distance').value,
                    np.array([point_dist[search_index - 1], point_dist[search_index]]),
                    np.array([path[search_index - 1, 0], path[search_index, 0]]))
                y_goal = np.interp(self.get_parameter('rrt_lookahead_distance').value,
                    np.array([point_dist[search_index - 1], point_dist[search_index]]),
                    np.array([path[search_index - 1, 1], path[search_index, 1]]))
                break
            else:
                x_goal = path[search_index, 0]
                y_goal = path[search_index, 1]
                break

        return x_goal, y_goal


    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood
    
    def transform_car_to_global(self, odom_msg, goal_x, goal_y):
        quaternion = [odom_msg.pose.pose.orientation.x,
                      odom_msg.pose.pose.orientation.y,
                      odom_msg.pose.pose.orientation.z,
                      odom_msg.pose.pose.orientation.w]

        rot_b_m = quaternion_matrix(quaternion)[:3, :3]
        trans_m = np.array(
            [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        tform_b_m = np.zeros((4, 4))
        tform_b_m[:3, :3] = rot_b_m
        tform_b_m[:3, 3] = trans_m
        tform_b_m[-1, -1] = 1
        goal_b = np.array([[goal_x],[goal_y], [0], [1]])
        goal_m = tform_b_m.dot(goal_b).flatten()
        return goal_m[0], goal_m[1]
        

    def transform_point(self, odom_msg, goalx, goaly):
        """
        World frame to vehicle frame
        """
        quaternion = [odom_msg.pose.pose.orientation.x,
                      odom_msg.pose.pose.orientation.y,
                      odom_msg.pose.pose.orientation.z,
                      odom_msg.pose.pose.orientation.w]

        rot_b_m = quaternion_matrix(quaternion)[:3, :3]
        trans_m = np.array(
            [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        tform_b_m = np.zeros((4, 4))
        tform_b_m[:3, :3] = rot_b_m
        tform_b_m[:3, 3] = trans_m
        tform_b_m[-1, -1] = 1

        goal_m = np.array([[goalx], [goaly], [0], [1]])
        goal_b = (np.linalg.inv(tform_b_m).dot(goal_m)).flatten()

        return goal_b[0], goal_b[1]

    def publish_occ_grid(self, pub, data, res, wh, origin=[0.0, 0.0, 0.0], frame='map'):
        """
        """
        # print(f"Publishing occupancy grid with origin x = {origin[0]}, y = {origin[1]}, z = {origin[2]}")
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.info.resolution = res
        msg.info.width = wh[0]
        msg.info.height = wh[1]
        msg.info.origin.position.x = origin[0]
        msg.info.origin.position.y = origin[1]
        quat = quaternion_from_euler(0.0, 0.0, origin[2])
        msg.info.origin.orientation.x = quat[0]
        msg.info.origin.orientation.y = quat[1]
        msg.info.origin.orientation.z = quat[2]
        msg.info.origin.orientation.w = quat[3]
        msg_data = -1 * np.ones(np.shape(data), dtype=int)
        msg_data[np.equal(data, 1)] = 0
        msg_data[np.equal(data, 0)] = 100
        msg.data = msg_data.flatten(order='F').tolist()
        pub.publish(msg)

    def publish_drive_msg(self, desired_angle):
        """
        """
        # Compute Control Input
        steering_angle_bound = 0.4
        angle = np.clip(desired_angle, -steering_angle_bound, steering_angle_bound)
        speed = np.interp(abs(angle), np.array([0.0, steering_angle_bound, np.inf]),
                          np.array([self.get_parameter('desired_speed').value, self.get_parameter('min_speed').value, self.get_parameter('min_speed').value]))

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = self.get_parameter('steering_angle_factor').value * angle
        msg.drive.speed = self.get_parameter('speed_factor').value * speed
        self.AckPublisher.publish(msg)

        return 0

    def generate_waypoint_path(self, sparse_points, waypoint_distance):
        """
        Callback for path service.
        """
        # Spline Interpolate Sparse Path
        # print("sparse_points = ")
        # print(sparse_points)
        tck, u = splprep(sparse_points.transpose(), s=0, per=True)
        approx_length = np.sum(np.linalg.norm(np.diff(splev(u, tck), axis=0), axis=1))
        # approx_length = np.sum(np.linalg.norm(
        #     np.diff(splev(np.linspace(0, 1, 100), tck), axis=0), axis=1))

        # Discretize Splined Path
        num_waypoints = int(approx_length / waypoint_distance)
        dense_points = splev(np.linspace(0, 1, num_waypoints), tck)
        dense_points = np.array([dense_points[0], dense_points[1], dense_points[2]]).transpose()

        # print("Sending Response")
        # print(response)
        return dense_points
    
    def publish_waypoint_msg(self, pub, xy, rgba=[255.0, 0.0, 0.0, 1.0], frame='map'):
        """
        """
        print(f"Publishing waypoint x = {xy[0]}, y = {xy[1]}, r = {rgba[0]}, g = {rgba[1]}, b = {rgba[2]}")
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = frame
        marker.type = marker.SPHERE
        marker.pose.position.x = xy[0]
        marker.pose.position.y = xy[1]
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]
        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        pub.publish(marker)
    
    def publish_waypoint_map_msg(self, pub, path, rgba=[255.0, 0.0, 0.0, 1.0], frame='map'):
        """
        """
        marker_array = MarkerArray()
        for idx, ps in enumerate(path):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = frame
            marker.id = idx
            marker.type = marker.SPHERE
            marker.pose.position.x = ps[0]
            marker.pose.position.y = ps[1]
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = rgba[0]
            marker.color.g = rgba[1]
            marker.color.b = rgba[2]
            marker.color.a = rgba[3]
            pt = Point()
            pt.x = marker.pose.position.x
            pt.y = marker.pose.position.y
            marker.points.append(pt)
            # marker.lifetime = rclpy.duration.Duration(seconds=0.5)
            marker_array.markers.append(marker)

        pub.publish(marker_array)
    
    def publish_dense_map(self):
        self.publish_waypoint_map_msg(self.WaypointMapvisualizer, self.path, [0.0, 255.0, 0.0, 1.0])

def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()