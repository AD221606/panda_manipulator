# Implementation Decisions

This document records the design decisions made while implementing the
perception and planning pipeline for the assignment. The implementation uses
only the three intended source files:

- `src/perception.py`
- `src/planning.py`
- `src/control.py`

The existing changes in `src/control.py` were retained and used as the
manipulation interface. The unrelated existing changes in `src/main.py` were
not modified.

## `src/perception.py`

### `ObjectDetector.__init__`

- The model path is constructed relative to `perception.py` using
  `os.path.dirname(__file__)`.
- This avoids depending on the process working directory. The program can
  therefore be launched from either the project root or another directory
  while still finding `src/weights/best.pt`.
- The provided YOLO weights are loaded once when the detector is created,
  rather than reloading them for every camera frame.

### `ObjectDetector.detect_objects`

- The sensor returns RGB images, while OpenCV and the detector expect BGR
  ordering. The image is therefore converted with
  `cv2.COLOR_RGB2BGR` before inference.
- YOLO inference uses `verbose=False` to avoid filling the assignment output
  with model logs.
- A confidence threshold of `0.25` was selected as a practical balance
  between accepting valid detections and rejecting weak proposals.
- The function returns a simple list of `(label, [x1, y1, x2, y2])` tuples.
  This keeps the detector independent from the rest of the pipeline and
  exposes only the information needed for cluster-to-detection matching.
- Bounding boxes are converted to CPU NumPy arrays of floating-point values,
  making them easy to compare with projected pixel coordinates.
- An empty result is returned when YOLO produces no result instead of raising
  an exception, because a valid camera frame may contain no detectable
  object.

### `Perception.__init__`

- The sensor, detector, manipulator, and depth processor are stored as
  collaborators so that each stage has one clear responsibility.
- The manipulator is injected rather than created inside `Perception`, which
  ensures that perception uses the same PyBullet client and Panda instance as
  planning and control.

### `Perception.generate_exploration_poses`

- The end-effector position is obtained from the current robot state, while
  only the camera orientation is swept. This produces viewpoints around the
  current workspace without assuming object locations.
- Yaw is sampled across `[-pi/2, pi/2]` using seven samples, giving a
  180-degree horizontal sweep.
- Pitch is sampled across `[-pi/6, pi/6]` using three samples, adding
  elevated and lowered views without creating an unnecessarily large scan.
- The end-effector roll is fixed at `pi` so the camera remains directed toward
  the workspace in the expected Panda-camera configuration.
- PyBullet inverse kinematics converts each Cartesian orientation into a
  seven-joint robot pose.
- Only the first seven values are retained because those are the Panda arm's
  revolute joints; any additional values returned by PyBullet are not arm
  joints used by `move_joints`.
- The sweep is deliberately compact rather than using a dense 5-degree
  yaw/pitch grid. This reduces motion time and the number of redundant point
  clouds while still providing multiple viewpoints.

### `Perception.identify_objects`

- Each point-cloud cluster is handled independently.
- The robot moves to the highest point of a cluster before taking an image.
  This provides a close, top-down view that improves the chance that the
  relevant object is visible to YOLO.
- The camera observation is obtained through the provided
  `RGBDSensor.get_observation(client, panda_id)` API rather than through
  privileged PyBullet object-position functions.
- The cluster centroid is projected into the current camera image. If that
  pixel lies inside a detection bounding box, that detection is assigned to
  the cluster.
- If the projected point is not inside any box, the closest detection-box
  center is used as a fallback. This handles small projection errors,
  imperfect cluster centroids, and partially occluded objects.
- Clusters with no detections are omitted instead of receiving a guessed
  semantic label.
- The robot returns home after processing each cluster. This reduces the
  chance of collision or an inconvenient starting configuration for the next
  observation.
- The result is a dictionary mapping labels to Open3D clusters, matching the
  data shape expected by the planner.

### `Perception.get_object_pick_pose`

- A missing object label raises `KeyError` rather than returning a fabricated
  pose. Invalid perception results should be visible to the caller.
- The highest cluster point, with its horizontal coordinates made robust by
  `get_cluster_top_position`, is used as the grasp location.
- A fixed top-down Euler orientation `[pi, 0, 0]` is returned because the
  assignment uses a crane-style grasp and the gripper should approach from
  above.

### `Perception.get_obj_place_pose`

- Both the destination and source labels are validated explicitly.
- The destination cluster's top position is used as the placement location.
  This makes placement depend on the sensed scene rather than hard-coded
  object coordinates.
- The same top-down orientation is used for placement to match the grasp
  orientation and the existing manipulator interface.
- The object label is checked even though its position is not needed to
  calculate the destination. This catches malformed plans early.

### `Perception.exlpore_scene`

- The method name retains the starter-code spelling `exlpore_scene` so the
  existing `main.py` call remains compatible.
- Each generated joint pose is sent through `move_joints`, then the RGB-D
  observation is captured after the motion.
- The sensor is called with the manipulator's actual PyBullet client and Panda
  ID, matching the implementation in `base.py`.
- RGB-D data is converted into a camera-frame point cloud and then transformed
  into the robot-base/world frame using the supplied camera transform.
- One point cloud is appended per successful exploration pose, preserving the
  relationship between observations and scan viewpoints.

### `DepthProcessor.__init__`

- No state is required by the current depth-processing operations, so the
  constructor remains intentionally empty.

### `DepthProcessor.rgbd_to_pointcloud`

- RGB and depth inputs are converted to NumPy arrays to support both regular
  arrays and array-like sensor outputs.
- Intrinsics are interpreted in the required order `[fx, fy, cx, cy]`.
- Pixel grids are generated vectorially with `np.indices` rather than using
  nested Python loops. This is substantially faster for a 640x480 image.
- The pinhole-camera equations from the assignment are used:
  `X=(u-cx)Z/fx`, `Y=(v-cy)Z/fy`, and `Z=depth`.
- Invalid, non-finite, near-zero, and out-of-range depths are removed. The
  limits `0.05` and `2.0` match the sensor's near/far operating range in
  `base.py`.
- RGB values are normalized to `[0, 1]`, which is the color convention
  expected by Open3D point clouds.
- The output is an Open3D colored `PointCloud`, allowing the same object to
  flow through registration, plane removal, and clustering.
- The requested width and height are clamped to the actual depth-array
  dimensions to avoid indexing errors if a test supplies a smaller image.

### `DepthProcessor.transform_pointcloud`

- The transform is converted to a floating-point NumPy array and validated
  as a 4x4 homogeneous transform.
- A copy of the input point cloud is transformed so callers do not have their
  original camera-frame data mutated unexpectedly.
- Open3D's built-in homogeneous transformation is used instead of manually
  rewriting points, preserving colors and Open3D metadata.

### `DepthProcessor.register_pointclouds`

- An empty input returns an empty Open3D point cloud instead of failing on
  index zero.
- The first point cloud is used as the initial reference frame.
- Point clouds are voxel-downsampled at `0.008` meters for registration. This
  reduces noise and computational cost while retaining object geometry.
- Normals are estimated only when at least three points are available, which
  avoids invalid normal-estimation calls on degenerate clouds.
- Each subsequent cloud is registered to the accumulated reference using
  point-to-plane ICP, an identity initial transform, and a correspondence
  distance of `2.5 * voxel_size`.
- The accumulated cloud is used as the next reference so all observations
  contribute to a common scene model.
- The final result is voxel-downsampled to reduce duplicate overlapping
  points.
- This registration is intentionally conservative because all observations
  have already been transformed using the known camera-to-base pose. ICP is
  used to correct small residual alignment errors rather than to solve an
  unconstrained global registration problem.

### `DepthProcessor.visualize_pointcloud`

- An empty list is ignored because there is nothing to display.
- Open3D's standard `draw_geometries` viewer is used so the method remains a
  simple optional debugging utility and does not affect the pipeline.

### `DepthProcessor.remove_plane`

- The input is voxel-downsampled before RANSAC to reduce runtime.
- Open3D `segment_plane` is used as required by the assignment.
- A distance threshold of `0.012` meters was chosen to tolerate depth noise
  while still separating tabletop points from raised objects.
- RANSAC uses three points per candidate plane and 1000 iterations, providing
  a stable estimate for a dominant tabletop or ground plane.
- The detected inlier plane is removed and the remaining point cloud is
  returned.
- If fewer than three points are available, the input is copied and returned
  because a plane cannot be estimated.

### `DepthProcessor.remove_planes`

- Plane removal is repeated up to three times to remove multiple dominant
  surfaces, such as a table and a ground-like support plane.
- The loop stops early when too few points remain or when a pass removes no
  points.
- The bounded iteration count prevents accidentally deleting arbitrary scene
  geometry if RANSAC repeatedly finds an unsuitable plane.

### `DepthProcessor.cluster_objects`

- Empty point clouds return an empty list.
- Open3D DBSCAN is used because the number of objects is not known in advance
  and clusters may have irregular shapes.
- `eps=0.025` meters defines the neighborhood radius and `min_points=20`
  rejects isolated depth noise.
- DBSCAN noise labels (`-1`) are discarded.
- Each remaining label is converted into an independent Open3D point cloud.
- Clusters with fewer than 20 points are ignored to prevent tiny noise
  fragments from becoming manipulable objects.

### `DepthProcessor.get_cluster_top_position`

- An empty cluster raises `ValueError` because no valid grasp position exists.
- The point with maximum world-frame Z is selected as the vertical top.
- The X and Y coordinates are replaced with the median X/Y of the cluster.
  This keeps the approach centered over the object while preserving the
  observed top height.
- The returned value is a regular Python list for compatibility with
  PyBullet inverse kinematics and the manipulator methods.

### `DepthProcessor.project_world_to_image`

- The world point is converted to homogeneous coordinates.
- The inverse of `T_cb` is applied because the supplied transform maps camera
  coordinates into the robot base/world frame.
- Points with non-positive camera-frame Z are returned as NaN pixel
  coordinates because they are behind the camera and cannot be projected.
- The standard pinhole projection equations are applied using the supplied
  intrinsics.
- Floating-point `(u, v)` values are returned rather than rounded integers,
  preserving precision for bounding-box comparisons.

## `src/planning.py`

### `Planner.__init__`

- A `Perception` instance is created with the same manipulator passed to the
  planner. This keeps pose lookup and robot motion connected to one
  simulation client.

### `Planner.get_clear_task_plan`

- The destination is selected from a sensed object label named `plate` or
  `shelf`, case-insensitively. Supporting both names accommodates the scene
  representation used by the assignment assets.
- A missing destination raises `ValueError` rather than generating a plan
  that cannot be executed.
- Every other detected object is converted into a pick-place subplan.
- The destination itself is skipped so the robot does not try to pick up the
  plate or shelf.
- A final home action is appended so the robot ends in a predictable safe
  configuration.
- Iteration follows dictionary insertion order, giving deterministic behavior
  for a fixed perception result without assuming object locations.

### `Planner.pick_place_object_plan`

- Pick and place poses are obtained from `Perception` rather than being
  recomputed or hard-coded in the planner.
- The plan uses dictionary action descriptors because
  `Manipulator.execute_task_plan` already supports that representation.
- Each subplan performs:
  1. pick the requested object,
  2. return home,
  3. place it at the destination,
  4. return home again.
- Returning home between phases reduces collision risk and gives each action a
  consistent starting configuration.

## `src/control.py`

The control implementation was already present when the perception and
planning code was completed. Its behavior was preserved rather than
duplicated:

### `Manipulator.is_moving`

- Reads the seven revolute-joint velocities and compares their maximum
  absolute value with `0.005`.
- This provides a simple convergence test for position-control commands.

### `Manipulator.move_joints`

- Uses PyBullet position control on the seven Panda revolute joints.
- Applies a force of `240.0` to each joint.
- Steps the simulation for at most 240 iterations and stops early once the
  joints are stable.

### `Manipulator.move_home`

- Reuses the Panda home pose defined by `PandaArm`, keeping home behavior
  centralized.

### `Manipulator.move_in_task_space`

- Accepts either Euler angles or quaternions to make callers flexible.
- Converts Euler orientations to quaternions before calling PyBullet inverse
  kinematics.
- Uses the Panda end-effector link and forwards only the seven arm joints to
  `move_joints`.

### `Manipulator.command_gripper`

- Uses the two linked Panda prismatic finger joints.
- Commands both fingers symmetrically and steps the simulation for 60
  iterations so the command has time to take effect.

### `Manipulator.pick_object_from`

- Uses a pre-grasp position 0.10 meters above the sensed object.
- Opens the gripper before approaching, lowers to the object, closes the
  gripper, and lifts back to the clearance height.
- This sequence avoids closing while the gripper is far from the object and
  avoids dragging the object horizontally.

### `Manipulator.place_object_at`

- Uses the same 0.10-meter clearance strategy for placement.
- Moves above the destination, lowers, opens the gripper, and retracts
  vertically.
- The release-before-retract order leaves the object at the target rather
  than carrying it away.

### `Manipulator.execute_task_plan`

- Supports dictionary action descriptors used by `Planner`.
- It also retains tuple/list action support from the starter implementation
  for compatibility with alternate plans.
- Recognized actions include pick, place, task-space motion, joint motion,
  home, and gripper commands.

## Validation Decisions

- Syntax was checked for all three permitted Python files using the
  assignment Conda environment.
- RGB-D conversion and identity point-cloud transformation were tested with
  synthetic data.
- Plane removal and DBSCAN clustering were tested with synthetic tabletop and
  raised-object points.
- The supplied YOLO model was loaded and run on a synthetic RGB image.
- The existing unrelated `src/main.py` modification was not reverted or
  changed.
