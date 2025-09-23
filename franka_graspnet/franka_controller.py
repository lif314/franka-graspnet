import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from franky import *

class FrankaController:
    def __init__(self, robot_ip: str = "192.168.1.1"):
        
        self.robot = Robot(robot_ip)

        # self.robot.relative_dynamics_factor = RelativeDynamicsFactor(0.05, 0.05, 0.05)
        self.robot.relative_dynamics_factor = 0.05

        # Set collision behavior
        lower_torque_thresholds = [20.0] * 7  # Nm
        upper_torque_thresholds = [40.0] * 7  # Nm
        lower_force_thresholds = [10.0] * 6  # N (linear) and Nm (angular)
        upper_force_thresholds = [20.0] * 6  # N (linear) and Nm (angular)
        self.robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        self.gripper = Gripper(robot_ip)

        self.HOME_JOINT_POSE = JointMotion(np.array([0. , -0.78539816,  0. , -2.35619449,  0. , 1.57079633,  0.78539816]))
        self.BOX_JOINT_POSE = JointMotion(np.array([1.12474915, -0.37809445, -0.04517126, -2.12450855,  0.00372194,  1.81733198, 0.50831428]))

        # Eye-in-hand
        self.R_Cam2Gripper = np.array([
            [ 0.0250142, -0.99918834, -0.03157457],
            [ 0.99874154,  0.02635147, -0.04267234],
            [ 0.04346974, -0.03046742,  0.99859006]
        ])
        self.T_Cam2Gripper = np.array([[0.05895725], [-0.02991709], [-0.03509327]])
        # EE 坐标系下的 Tool 偏移
        self.TOOL_IN_EE = np.array([-0.010, -0.000, 0.080]) # Z越小，夹爪越向下

    def move_home(self):
        self.robot.move(self.HOME_JOINT_POSE)

    def move_box(self):
        self.robot.move(self.BOX_JOINT_POSE)

    def open_gripper(self, speed=0.04):
        self.gripper.open(speed=speed)

    def close_gripper(self, width=0.0, speed=0.04, force=80):
        self.gripper.grasp(width, speed=speed, force=force)

    def compute_target_pose(self, target_gg):
        """根据 GraspNet 预测和相机外参计算 Base 下目标位姿"""
        # Grasp pose in camera
        R_grasp2camera = target_gg.rotation_matrix
        t_grasp2camera = target_gg.translation.reshape(3, 1)

        # Camera -> EE
        point_ee = self.R_Cam2Gripper @ t_grasp2camera + self.T_Cam2Gripper
        point_ee = point_ee.flatten()

        # EE pose in Base
        ee_pose_base = self.robot.current_cartesian_state.pose.end_effector_pose
        R_ee_base = ee_pose_base.matrix[:3, :3]

        # EE -> Base
        point_ee_in_base = ee_pose_base * Affine(point_ee, np.array([0.0, 0.0, 0.0, 1.0]))
        point_ee_in_base_pos = point_ee_in_base.translation

        # Tool 偏移在 Base 下
        tool_in_base = R_ee_base @ self.TOOL_IN_EE
        ee_target_in_base = point_ee_in_base_pos - tool_in_base

        # -----------------------------
        # 构造 EE 旋转矩阵
        # -----------------------------
        approach = R_grasp2camera[:, 0]  # GraspNet x = approach
        open_dir = R_grasp2camera[:, 1]  # GraspNet y = open
        z_ee = approach
        y_ee = open_dir
        x_ee = np.cross(y_ee, z_ee)

        R_target_ee = np.column_stack([x_ee, y_ee, z_ee])
        R_target_ee = self.R_Cam2Gripper @ R_target_ee
        R_target_base = R_ee_base @ R_target_ee

        # === 保证 z 轴朝上，消除180度的二义性 ===
        if R_target_base[2, 2] < 0:  # 如果 z 轴朝下
            R_target_base = R_target_base @ R.from_euler("z", 180, degrees=True).as_matrix()

        q_target_base = R.from_matrix(R_target_base).as_quat()

        return Affine(ee_target_in_base, q_target_base)

    def execute_grasp(self, target_pose_base):
        """执行一次完整抓取：移动 -> 闭合 -> 放置 -> 回 home"""
        try:
            if self.gripper.width < 0.01:
                self.open_gripper()

            self.robot.move(CartesianMotion(target_pose_base, ReferenceType.Absolute))
            self.close_gripper(width=0.0, speed=0.04, force=80)
            self.move_box()

            self.open_gripper()
            time.sleep(1.0)
            self.move_home()

        except Exception as e:
            print("抓取执行出错：", e)
