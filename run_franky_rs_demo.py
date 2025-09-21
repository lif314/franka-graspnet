import os
import sys
import numpy as np
import open3d as o3d
import cv2
import time
import scipy.ndimage as nd
import torch
from scipy.spatial.transform import Rotation as R
from graspnetAPI import GraspGroup
from franka_graspnet.realsense import RGBDCamera, get_devices

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default="checkpoints/graspnet/checkpoint-rs.tar", help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000)
parser.add_argument('--num_view', type=int, default=300)
parser.add_argument('--collision_thresh', type=float, default=0.01)
parser.add_argument('--voxel_size', type=float, default=0.01)
cfgs = parser.parse_args()


# Robot config
from franky import *

# Eye-in-hand
R_Cam2Gripper = np.array([
    [ 0.0250142, -0.99918834, -0.03157457],
    [ 0.99874154,  0.02635147, -0.04267234],
    [ 0.04346974, -0.03046742,  0.99859006]
])
T_Cam2Gripper = np.array([[0.05895725], [-0.02991709], [-0.03509327]])
# EE 坐标系下的 Tool 偏移
TOOL_IN_EE = np.array([-0.010, -0.000, 0.085]) # Z越小，夹爪约向下

ROBOT_IP = "192.168.1.1"
JOINT_POSITION_START = np.array([0. , -0.78539816,  0. , -2.35619449,  0. , 1.57079633,  0.78539816])
JOINT_POSITION_BOX = np.array([1.12474915, -0.37809445, -0.04517126, -2.12450855,  0.00372194,  1.81733198, 0.50831428])
HOME_JOINT_POSE = JointMotion(JOINT_POSITION_START)
BOX_JOINT_POSE = JointMotion(JOINT_POSITION_BOX)

def get_net():
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print("-> loaded checkpoint", cfgs.checkpoint_path)
    return net

def process_realsense_data(rgb_image, depth_image, rgb_intrinsics, workspace_mask=None,
                           plane_remove=True, plane_dist_thresh=0.2):
    """
    将 Realsense 实时 RGB-D 图像转换成 GraspNet 输入
    并可选集成 RANSAC 平面去噪
    """
    fx, fy = rgb_intrinsics.fx, rgb_intrinsics.fy
    cx, cy = rgb_intrinsics.ppx, rgb_intrinsics.ppy
    camera_info = CameraInfo(
        width=rgb_image.shape[1],
        height=rgb_image.shape[0],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        scale=1.0
    )

    # 1. 生成点云
    cloud = create_point_cloud_from_depth_image(depth_image, camera_info, organized=True)

    if workspace_mask is None:
        mask = depth_image > 0
    else:
        mask = (depth_image > 0) & (workspace_mask > 0)

    cloud_masked = cloud[mask]
    color_masked = rgb_image[mask]

    # 2. 平面过滤 (RANSAC)
    if plane_remove and len(cloud_masked) > 100:
        cloud_o3d_tmp = o3d.geometry.PointCloud()
        cloud_o3d_tmp.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        plane_model, inliers = cloud_o3d_tmp.segment_plane(distance_threshold=plane_dist_thresh,
                                                           ransac_n=3,
                                                           num_iterations=1000)
        inliers = np.array(inliers)
        # 保留靠近平面的 inliers，剔除远点
        cloud_masked = cloud_masked[inliers]
        color_masked = color_masked[inliers]

    # 3. 采样
    num_point = cfgs.num_point
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # 4. Open3D 点云 (可视化用)
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # 5. 转 torch (GraspNet 输入)
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled
    return end_points, cloud_o3d

def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    return GraspGroup(gg_array)


def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(np.array(cloud.points), voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud, num=50):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:num]

    grippers = gg.to_open3d_geometry_list()

    T = np.diag([1, 1, -1, 1])
    cloud.transform(T)

    for g in grippers:
        g.transform(T)

    o3d.visualization.draw_geometries([cloud, *grippers])


def create_rect_mask(image_shape, top_left, bottom_right):
    """
    创建矩形区域 mask
    :param image_shape: (H, W) 与 depth/rgb 图像一致
    :param top_left: (x1, y1) 左上角坐标
    :param bottom_right: (x2, y2) 右下角坐标
    :return: mask, dtype=bool
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    x1, y1 = top_left
    x2, y2 = bottom_right

    # 限制边界范围
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    mask[y1:y2, x1:x2] = True
    return mask

def filter_by_vertical_angle(grasp_group, angle_threshold_deg=30):
    """根据垂直角度筛选抓取"""
    all_grasps = list(grasp_group)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
    filtered = []
    
    for grasp in all_grasps:
        # 抓取的接近方向
        approach_dir = grasp.rotation_matrix[:, 0]
        # 计算与垂直方向的夹角
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        angle_threshold = np.deg2rad(angle_threshold_deg)
        if angle < angle_threshold:
            filtered.append(grasp)
    
    if len(filtered) == 0:
        print(f"\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    else:
        print(f"\nFiltered {len(filtered)} grasps within ±{np.rad2deg(angle_threshold):.0f}° of vertical out of {len(all_grasps)} total predictions.")
    
    return filtered

def demo_realsense():
    print("Init realsense camera.......")
    device_serials = get_devices()
    if len(device_serials) == 0:
        print("No Realsense devices found!")
        return
    print("Selected device serial numbers:", device_serials[0])
    rgb_resolution = (1280, 720)
    depth_resolution = (1280, 720)
    camera = RGBDCamera(device_serials[0], rgb_resolution, depth_resolution)
    camera.start()
    rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs = camera.get_intrinsics_raw()
    depth_scale = camera.get_depth_scale()
    print("depth_scale:", depth_scale)

    # ROBOT INIT
    fr3_robot = Robot(ROBOT_IP)
    fr3_robot.relative_dynamics_factor = 0.05
    fr3_gripper = Gripper(ROBOT_IP)
    print("Move robot to start position...")
    fr3_robot.move(HOME_JOINT_POSE)
    # if fr3_gripper.width < 0.01:
    fr3_gripper.open(speed=0.04)

    net = get_net()


    for _ in range(5):
        _, _, = camera.shoot()

    workspace_mask = create_rect_mask((720, 1280), (200, 0), (1150, 600))

    print("Start real-time grasp prediction...")
    try:
        while True:
            rgb_image, depth_image = camera.shoot()
            rgb_image = rgb_image / 255.0  # RGB
            depth_image = depth_image * depth_scale
            # depth filter
            depth_image[depth_image > 1.2] = 0

            end_points, cloud = process_realsense_data(rgb_image, depth_image, rgb_intrinsics, workspace_mask)
            gg = get_grasps(net, end_points)
            if cfgs.collision_thresh > 0:
                gg = collision_detection(gg, cloud)

            if len(gg) == 0:
                print("Not detected grasps...")
                time.sleep(1.0)
                continue

            vis_grasps(gg, cloud, num=50)

            # Robot grasp
            gg.nms()
            gg.sort_by_score()
            # 垂直角度筛选
            filtered_grasps = filter_by_vertical_angle(gg)

            target_gg = filtered_grasps[0]
            print("Target grasp pose: ", target_gg)
            """
            Target grasp pose:  Grasp: score:0.10523781925439835, width:0.08736982941627502, height:0.019999999552965164, depth:0.029999999329447746, translation:[-0.374148   -0.24019353  0.629     ]
                rotation:
                [[ 0.07559296  0.01005337  0.9970881 ]
                [ 0.28120205 -0.9595779  -0.0116438 ]
                [ 0.95666665  0.2812634  -0.07536437]]
            """

            # 取 target grasp 的旋转和平移
            R_grasp2camera = target_gg.rotation_matrix
            t_grasp2camera = target_gg.translation.reshape(3,1) # [x, y, z]

            # Camera -> EE -> Base
            point_ee = R_Cam2Gripper @ t_grasp2camera + T_Cam2Gripper
            point_ee = point_ee.flatten()  # EE 坐标系下目标点

            ee_pose_base = fr3_robot.current_cartesian_state.pose.end_effector_pose  # Affine
            R_ee_base = ee_pose_base.matrix[:3, :3]
            q_ee_pose_base = ee_pose_base.quaternion

            # EE -> Base
            point_ee_in_base = ee_pose_base * Affine(point_ee, np.array([0.0, 0.0, 0.0, 1.0]))
            point_ee_in_base_pos = point_ee_in_base.translation

            # Tool 偏移在 Base 下
            tool_in_base = R_ee_base @ TOOL_IN_EE
            ee_target_in_base = point_ee_in_base_pos - tool_in_base

            # -----------------------------
            # 旋转矩阵处理
            # -----------------------------
            # GraspNet axes (camera): x=approach, y=open, z=垂直
            # Franka EE axes: z=approach, y=open, x=垂直
            """
            保证右手系
                GraspNet 输出：
                    x = approach
                    y = gripper open
                    z = perpendicular
                EE 末端 期望：
                    z = approach
                    y = gripper open
                    x = z × y （通过叉乘保证右手系）
            """
            approach = R_grasp2camera[:, 0]  # x axis
            open_dir = R_grasp2camera[:, 1]  # y axis
            z_ee = approach                 # EE z-axis = approach
            y_ee = open_dir                 # EE y-axis = gripper open
            x_ee = np.cross(y_ee, z_ee)    # EE x-axis = y × z, 保证右手系

            # 构造 EE 旋转矩阵
            R_target_ee = np.column_stack([x_ee, y_ee, z_ee])
            R_target_ee = R_Cam2Gripper @ R_target_ee  # Camera -> EE
            R_target_base = R_ee_base @ R_target_ee
            q_target_base = R.from_matrix(R_target_base).as_quat()


            # 构造 Base 下目标位姿
            target_pose_base = Affine(ee_target_in_base, q_target_base)

            # 执行抓取
            try:
                if fr3_gripper.width < 0.01:
                    fr3_gripper.open(speed=0.04)

                fr3_robot.move(CartesianMotion(target_pose_base, ReferenceType.Absolute))
                fr3_gripper.grasp(0.00, speed=0.04, force=80)
                fr3_robot.move(BOX_JOINT_POSE)

                fr3_gripper.open(speed=0.04)
                time.sleep(1.0)
                fr3_robot.move(HOME_JOINT_POSE)

            except Exception as e:
                print("抓取执行出错：", e)
    except KeyboardInterrupt:
        print("Exit demo.")
        camera.stop()


if __name__ == "__main__":
    demo_realsense()
