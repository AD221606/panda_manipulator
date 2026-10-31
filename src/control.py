import math
import time
import numpy as np
from base import PandaArm


class Manipulator(PandaArm):

    def __init__(self, p, pandaId):
        super().__init__(p, pandaId)

    def is_moving(self, threshold: float = 0.005) -> bool:
        """Checks if any arm joint velocity exceeds the movement threshold."""
        joint_states = self.pybullet_client.getJointStates(
            self.pandaId, self.REVOLUTE_JOINTS
        )
        velocities = [state[1] for state in joint_states]
        return bool(np.max(np.abs(velocities)) > threshold)

    def move_joints(
        self,
        target_joints: list,
        max_steps: int = 240,
        wait_for_convergence: bool = True,
    ):
        """Moves manipulator joints to target joint poses (radians)."""
        self.pybullet_client.setJointMotorControlArray(
            bodyUniqueId=self.pandaId,
            jointIndices=self.REVOLUTE_JOINTS,
            controlMode=self.pybullet_client.POSITION_CONTROL,
            targetPositions=target_joints,
            forces=[240.0] * 7,
        )

        if wait_for_convergence:
            for _ in range(max_steps):
                self.pybullet_client.stepSimulation()
                # Early stop if arm has stabilized at target
                if not self.is_moving():
                    break

    def move_home(self):
        """Moves robot to pre-defined home position."""
        self.move_joints(self.PANDA_HOME_POSITION)

    def move_in_task_space(self, target_position: list, target_orientation: list):
        """Moves robot end-effector to Cartesian target pose.

        Accepts target_orientation in either Euler angles [roll, pitch, yaw] or
        Quaternion [x, y, z, w].
        """
        p = self.pybullet_client

        # Convert Euler angles to Quaternion if necessary
        if len(target_orientation) == 3:
            target_orn_quat = p.getQuaternionFromEuler(target_orientation)
        else:
            target_orn_quat = target_orientation

        joint_angles = p.calculateInverseKinematics(
            bodyUniqueId=self.pandaId,
            endEffectorLinkIndex=self.PANDA_END_EFFECTOR_INDEX,
            targetPosition=target_position,
            targetOrientation=target_orn_quat,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )

        self.move_joints(joint_angles[:7])

    def command_gripper(self, open: bool) -> bool:
        """Opens or closes robot's prismatic gripper fingers."""
        p = self.pybullet_client
        target_pos = (
            self.PANDA_FINGER_OPEN if open else self.PANDA_FINGER_CLOSED
        )

        p.setJointMotorControlArray(
            bodyUniqueId=self.pandaId,
            jointIndices=self.PANDA_PRISMATIC_JOINTS,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target_pos, target_pos],
            forces=[20.0, 20.0],
        )

        for _ in range(60):
            p.stepSimulation()

        return True

    def pick_object_from(
        self,
        obj_location: list,
        obj_orientation: list,
        clearance: float = 0.10,
    ):
        """Performs structured pick operation (Pre-grasp -> Lower -> Close -> Lift)."""
        pre_grasp_pos = [
            obj_location[0],
            obj_location[1],
            obj_location[2] + clearance,
        ]

        # 1. Open gripper & move to pre-grasp pose above object
        self.command_gripper(open=True)
        self.move_in_task_space(pre_grasp_pos, obj_orientation)

        # 2. Lower to object target position
        self.move_in_task_space(obj_location, obj_orientation)

        # 3. Close gripper to grasp
        self.command_gripper(open=False)

        # 4. Lift up back to pre-grasp clearance
        self.move_in_task_space(pre_grasp_pos, obj_orientation)

    def place_object_at(
        self,
        target_location: list,
        target_orientation: list,
        clearance: float = 0.10,
    ):
        """Performs structured place operation (Pre-place -> Lower -> Open -> Retract)."""
        pre_place_pos = [
            target_location[0],
            target_location[1],
            target_location[2] + clearance,
        ]

        # 1. Move to pre-place position above target
        self.move_in_task_space(pre_place_pos, target_orientation)

        # 2. Lower object onto target position
        self.move_in_task_space(target_location, target_orientation)

        # 3. Open gripper to release object
        self.command_gripper(open=True)

        # 4. Retract straight up
        self.move_in_task_space(pre_place_pos, target_orientation)

    def execute_task_plan(self, plan: list):
        """Executes sequential task plan steps (supports dict or tuple action descriptors)."""
        for step in plan:
            if isinstance(step, dict):
                action = step.get("action")
                if action == "pick":
                    self.pick_object_from(
                        step["location"], step["orientation"]
                    )
                elif action == "place":
                    self.place_object_at(step["location"], step["orientation"])
                elif action == "move_task":
                    self.move_in_task_space(
                        step["position"], step["orientation"]
                    )
                elif action == "move_joints":
                    self.move_joints(step["joints"])
                elif action == "home":
                    self.move_home()
                elif action == "gripper":
                    self.command_gripper(step["open"])

            elif isinstance(step, (tuple, list)):
                action_name = step[0]
                if action_name == "pick":
                    self.pick_object_from(step[1], step[2])
                elif action_name == "place":
                    self.place_object_at(step[1], step[2])
                elif action_name == "home":
                    self.move_home()
                elif action_name == "gripper":
                    self.command_gripper(step[1])