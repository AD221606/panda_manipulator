from base import RGBDSensor
from control import Manipulator

from ultralytics import YOLO
import open3d as o3d
import numpy as np
import cv2


class ObjectDetector:
    def __init__(self,):
        self.detector = YOLO("./weights/best.pt")

    def detect_objects(self, rgbImage):
        '''
        returns obj_labels and 2D bounding box the objects in the rgb image
        args
            rgbImage : RGB image containing object to be identified

        please note that due to openCV default BGR convention, you will need to 
        first convert the image to BGR before feeding to the detector
        '''
        # TODO


class Perception:
    def __init__(self, manipulator:Manipulator):
        self.sensor = RGBDSensor()
        self.obj_detector = ObjectDetector()
        self.manipulator = manipulator
        self.depth_processor = DepthProcessor()

    def generate_exploration_poses(self,):
        '''
        returns list of 7D joint poses to explore the scene
        '''
        # TODO

    def identify_objects(self, pointcloud_clusters):
        '''
        assigns labels to all object clusters in the pointcloud_clusters
        args
            pointcloud_clusters : clustered pointclouds
        returns
            objs : dictionary containing labels and spatial information of all the objects in the scene
                    {obj_label: obj_pointcloud}
        '''
        # TODO

    def get_object_pick_pose(self, obj_label:str, objs:dict):
        '''
        returns 6D pose of the queried object: obj_label
        args
            obj_label : query object label
            objs : dictionary containing labels and spatial information of all the objects in the scene
        '''
        # TODO
    
    def get_obj_place_pose(self, obj_label:str, target_obj_label:str, objs:dict):
        '''
        returns 6D target pose for object to be placed at target object location
        args
            obj_label : to be placed object label
            target_obj_label : label of the object on which object is to be placed
            objs : dictionary containing labels and spatial information of all the objects in the scene
        '''
        # TODO
    
    def exlpore_scene(self, exlporation_poses:list[list]):
        '''
        explore the scene using generated exploration poses and return collected pointclouds
        args
            exploration_poses : list with set of proposed 7D joint positions fullfilling environment/scene exploration
        returns
            pointclouds : list of collected pointclouds from the exploration
        '''
        # TODO


class DepthProcessor:
    def __init__(self,):
        pass

    def rgbd_to_pointcloud(self, rgb, depth, intrinsics, width, height):
        '''
        returns a colored pointcloud in the camera frame
        args
            rgb : RGB image
            depth : depth image
            intrinsics : list - [fx, fy, cx, cy] : intrinsic parameters from k matrix
        '''
        # TODO

    def transform_pointcloud(self, pointcloud:o3d.geometry.PointCloud, transform):
        '''
        returns a pointcloud transformed in the target transform frame
        args
            pointcloud : 
            transform : 
        returns
            pointcloud_base : pointcloud transformed in the robot base frame
        '''
        # TODO

    def register_pointclouds(self, pcds:list):
        '''
        returns a poincloud aligned in a single frame
        args
            pcds : list of pointclouds
        returns
            scene_pointcloud : combined and aligned pointcloud in a single frame from pcds
        '''
        # TODO

    def visualize_pointcloud(self, pcds:list[o3d.geometry.PointCloud]):
        '''
        this function will visualize the pointcloud(s) present in the list `pcds`
        args
            pcds: list of pointclouds to be visualized (could be single/multiple pointclouds)
        '''
        # TODO

    def remove_plane(self, scene_pcd):
        '''
        this function removes a plane that maximally fits a plane equation (criterion)
        args
            scene_pcd : pointcloud of a scene
        '''
        # TODO

    def remove_planes(self, scene_pcd):
        '''
        this function iteratively removes multiple planes from the scene_pcd
        '''
        # TODO

    def cluster_objects(self, scene_pcd):
        '''
        this function clusters the pointcloud
        args
            scene_pcd : merged pointcloud of the scene
        '''
        # TODO

    def get_cluster_top_position(self, cluster):
        '''
        args
            cluster : a cluster from the clustered pointcloud which manipulator aims to understand
        '''
        # TODO
    
    def project_world_to_image(self, point_world, intrinsics, T_cb):
        """
        args:
            point_world : (x,y,z) in robot base (world) frame
            intrinsics : list [fx, fy, cx, cy] - camera intrinsic parameters
            T_cb: (4x4) transform of camera frame wrt robot base frame
        returns:
            (u, v): pixel coordinates
        """
        # TODO
        return (u, v)
    