from base import RGBDSensor
from control import Manipulator
import math
import os
from ultralytics import YOLO
import open3d as o3d
import numpy as np
import cv2


class ObjectDetector:
    def __init__(self,):
        weights_path = os.path.join(
            os.path.dirname(__file__), "weights", "best.pt"
        )
        self.detector = YOLO(weights_path)

    def detect_objects(self, rgbImage):
        '''
        returns obj_labels and 2D bounding box the objects in the rgb image
        args
            rgbImage : RGB image containing object to be identified

        please note that due to openCV default BGR convention, you will need to 
        first convert the image to BGR before feeding to the detector
        '''
        image_bgr = cv2.cvtColor(np.asarray(rgbImage), cv2.COLOR_RGB2BGR)
        results = self.detector.predict(
            source=image_bgr, verbose=False, conf=0.25
        )
        detections = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = names[class_id]
            coordinates = box.xyxy[0].cpu().numpy().astype(float)
            detections.append((label, coordinates))
        return detections


class Perception:
    def __init__(self, manipulator:Manipulator):
        self.sensor = RGBDSensor()
        self.obj_detector = ObjectDetector()
        self.manipulator = manipulator
        self.depth_processor = DepthProcessor()

    def generate_exploration_poses(self):
        '''
        returns list of 7D joint poses to explore the scene
        '''
        p = self.manipulator.pybullet_client
        panda_id = self.manipulator.pandaId
        ee_index = self.manipulator.PANDA_END_EFFECTOR_INDEX
        current_pos = p.getLinkState(
            panda_id, ee_index, computeForwardKinematics=True
        )[4]
        poses = []
        
        for yaw in np.linspace(-math.pi / 2, math.pi / 2, 7):
            for pitch in np.linspace(-math.pi / 6, math.pi / 6, 3):
                target_orn = p.getQuaternionFromEuler(
                    [math.pi, pitch, yaw]
                )
                full_ik_solution = p.calculateInverseKinematics(
                    bodyUniqueId=panda_id,
                    endEffectorLinkIndex=ee_index,
                    targetPosition=current_pos,
                    targetOrientation=target_orn,
                    maxNumIterations=200,
                    residualThreshold=1e-5
                )

                
                poses.append(list(full_ik_solution[:7]))

        return poses


    def identify_objects(self, pointcloud_clusters):
        '''
        assigns labels to all object clusters in the pointcloud_clusters
        args
            pointcloud_clusters : clustered pointclouds
        returns
            objs : dictionary containing labels and spatial information of all the objects in the scene
                    {obj_label: obj_pointcloud}
        '''
        objs = {}
        robot_base = self.manipulator.pybullet_client
        for cluster in pointcloud_clusters:
            top_position = self.depth_processor.get_cluster_top_position(cluster)
            self.manipulator.move_in_task_space(
                top_position, [math.pi, 0.0, 0.0]
            )
            rgb_image, _, transform, intrinsics = self.sensor.get_observation(
                robot_base, self.manipulator.pandaId
            )
            detections = self.obj_detector.detect_objects(rgb_image)
            center = np.asarray(cluster.get_center())
            pixel = self.depth_processor.project_world_to_image(
                center, intrinsics, transform
            )

            matching_label = None
            for label, box in detections:
                if box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3]:
                    matching_label = label
                    break
            if matching_label is None and detections:
                matching_label = min(
                    detections,
                    key=lambda detection: np.linalg.norm(
                        np.asarray(
                            [
                                (detection[1][0] + detection[1][2]) / 2,
                                (detection[1][1] + detection[1][3]) / 2,
                            ]
                        )
                        - np.asarray(pixel)
                    ),
                )[0]
            if matching_label is not None:
                objs[matching_label] = cluster
            self.manipulator.move_home()
        return objs

    def get_object_pick_pose(self, obj_label:str, objs:dict):
        '''
        returns 6D pose of the queried object: obj_label
        args
            obj_label : query object label
            objs : dictionary containing labels and spatial information of all the objects in the scene
        '''
        if obj_label not in objs:
            raise KeyError(f"Object {obj_label!r} was not detected")
        return self.get_cluster_top_position(objs[obj_label]), [math.pi, 0, 0]
    
    def get_obj_place_pose(self, obj_label:str, target_obj_label:str, objs:dict):
        '''
        returns 6D target pose for object to be placed at target object location
        args
            obj_label : to be placed object label
            target_obj_label : label of the object on which object is to be placed
            objs : dictionary containing labels and spatial information of all the objects in the scene
        '''
        if target_obj_label not in objs:
            raise KeyError(f"Target {target_obj_label!r} was not detected")
        target = np.asarray(self.get_cluster_top_position(objs[target_obj_label]))
        if obj_label not in objs:
            raise KeyError(f"Object {obj_label!r} was not detected")
        return target.tolist(), [math.pi, 0, 0]
    
    def exlpore_scene(self, exlporation_poses:list[list]):
        '''
        explore the scene using generated exploration poses and return collected pointclouds
        args
            exploration_poses : list with set of proposed 7D joint positions fullfilling environment/scene exploration
        returns
            pointclouds : list of collected pointclouds from the exploration
        '''
        collected_pointclouds = []
        for p in exlporation_poses:
            self.manipulator.move_joints(p)
            rgbImage, depthImage, T_cb, intrinsics = self.sensor.get_observation(
                self.manipulator.pybullet_client, self.manipulator.pandaId
            )
            pointcloud = self.depth_processor.rgbd_to_pointcloud(
                rgbImage,
                depthImage,
                intrinsics,
                self.sensor.IMG_WIDTH,
                self.sensor.IMG_HEIGHT,
            )
            pointcloud = self.depth_processor.transform_pointcloud(
                pointcloud, T_cb
            )
            collected_pointclouds.append(pointcloud)

        return collected_pointclouds


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
        rgb = np.asarray(rgb)
        depth = np.asarray(depth, dtype=np.float32)
        fx, fy, cx, cy = intrinsics
        height = min(height, depth.shape[0])
        width = min(width, depth.shape[1])
        v, u = np.indices((height, width))
        z = depth[:height, :width]
        valid = np.isfinite(z) & (z > 0.05) & (z < 2.0)
        points = np.column_stack(
            (
                (u[valid] - cx) * z[valid] / fx,
                (v[valid] - cy) * z[valid] / fy,
                z[valid],
            )
        )
        colors = rgb[:height, :width][valid].astype(np.float64) / 255.0
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return pointcloud

    def transform_pointcloud(self, pointcloud:o3d.geometry.PointCloud, transform):
        '''
        returns a pointcloud transformed in the target transform frame
        args
            pointcloud : 
            transform : 
        returns
            pointcloud_base : pointcloud transformed in the robot base frame
        '''
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("Point-cloud transform must have shape (4, 4)")
        transformed = o3d.geometry.PointCloud(pointcloud)
        transformed.transform(transform)
        return transformed

    def register_pointclouds(self, pcds:list):
        '''
        returns a poincloud aligned in a single frame
        args
            pcds : list of pointclouds
        returns
            scene_pointcloud : combined and aligned pointcloud in a single frame from pcds
        '''
        if not pcds:
            return o3d.geometry.PointCloud()
        merged = o3d.geometry.PointCloud(pcds[0])
        voxel_size = 0.008
        reference = merged.voxel_down_sample(voxel_size)
        if len(reference.points) >= 3:
            reference.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )
        for source in pcds[1:]:
            source_copy = o3d.geometry.PointCloud(source)
            source_down = source_copy.voxel_down_sample(voxel_size)
            if len(reference.points) >= 3 and len(source_down.points) >= 3:
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size * 2, max_nn=30
                    )
                )
                result = o3d.pipelines.registration.registration_icp(
                    source_down,
                    reference,
                    voxel_size * 2.5,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )
                source_copy.transform(result.transformation)
            merged += source_copy
            reference = merged.voxel_down_sample(voxel_size)
            if len(reference.points) >= 3:
                reference.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size * 2, max_nn=30
                    )
                )
        return merged.voxel_down_sample(voxel_size)

    def visualize_pointcloud(self, pcds:list[o3d.geometry.PointCloud]):
        '''
        this function will visualize the pointcloud(s) present in the list `pcds`
        args
            pcds: list of pointclouds to be visualized (could be single/multiple pointclouds)
        '''
        if not pcds:
            return
        o3d.visualization.draw_geometries(pcds)

    def remove_plane(self, scene_pcd):
        '''
        this function removes a plane that maximally fits a plane equation (criterion)
        args
            scene_pcd : pointcloud of a scene
        '''
        downsampled = scene_pcd.voxel_down_sample(0.005)
        if len(downsampled.points) < 3:
            return o3d.geometry.PointCloud(scene_pcd)
        _, inliers = downsampled.segment_plane(
            distance_threshold=0.012, ransac_n=3, num_iterations=1000
        )
        return downsampled.select_by_index(inliers, invert=True)

    def remove_planes(self, scene_pcd):
        '''
        this function iteratively removes multiple planes from the scene_pcd
        '''
        remaining = o3d.geometry.PointCloud(scene_pcd)
        for _ in range(3):
            if len(remaining.points) < 3:
                break
            candidate = self.remove_plane(remaining)
            if len(candidate.points) == len(remaining.points):
                break
            remaining = candidate
        return remaining

    def cluster_objects(self, scene_pcd):
        '''
        this function clusters the pointcloud
        args
            scene_pcd : merged pointcloud of the scene
        '''
        if len(scene_pcd.points) == 0:
            return []
        labels = np.asarray(
            scene_pcd.cluster_dbscan(
                eps=0.025, min_points=20, print_progress=False
            )
        )
        clusters = []
        for label in sorted(set(labels)):
            if label < 0:
                continue
            indices = np.flatnonzero(labels == label)
            cluster = scene_pcd.select_by_index(indices)
            if len(cluster.points) >= 20:
                clusters.append(cluster)
        return clusters

    def get_cluster_top_position(self, cluster):
        '''
        args
            cluster : a cluster from the clustered pointcloud which manipulator aims to understand
        '''
        points = np.asarray(cluster.points)
        if points.size == 0:
            raise ValueError("Cannot find a top position for an empty cluster")
        highest = np.argmax(points[:, 2])
        top = points[highest].copy()
        top[:2] = np.median(points[:, :2], axis=0)
        return top.tolist()
    
    def project_world_to_image(self, point_world, intrinsics, T_cb):
        """
        args:
            point_world : (x,y,z) in robot base (world) frame
            intrinsics : list [fx, fy, cx, cy] - camera intrinsic parameters
            T_cb: (4x4) transform of camera frame wrt robot base frame
        returns:
            (u, v): pixel coordinates
        """
        point_camera = np.linalg.inv(np.asarray(T_cb)) @ np.r_[
            np.asarray(point_world, dtype=float), 1.0
        ]
        if point_camera[2] <= 0:
            return (float("nan"), float("nan"))
        fx, fy, cx, cy = intrinsics
        u = fx * point_camera[0] / point_camera[2] + cx
        v = fy * point_camera[1] / point_camera[2] + cy
        return (float(u), float(v))
    