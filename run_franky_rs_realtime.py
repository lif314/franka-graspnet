import argparse
import numpy as np
import open3d as o3d

from franka_graspnet.franka_controller import FrankaController
from franka_graspnet.graspnet_infer import GraspNetInfer
from franka_graspnet.realsense import RGBDCamera, get_devices

def create_rect_mask(image_shape, top_left, bottom_right):
    """创建矩形区域 mask"""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    mask[y1:y2, x1:x2] = True
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/graspnet/checkpoint-rs.tar")
    parser.add_argument('--num_point', type=int, default=20000)
    parser.add_argument('--num_view', type=int, default=300)
    parser.add_argument('--collision_thresh', type=float, default=0.01)
    parser.add_argument('--angle_threshold_deg', type=float, default=30)
    parser.add_argument('--voxel_size', type=float, default=0.01)

    # for camera
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    cfgs = parser.parse_args()

    # === Camera Init ===
    print("Init realsense camera.......")
    device_serials = get_devices()
    if len(device_serials) == 0:
        print("No Realsense devices found!")
        return
    print("Selected device serial:", device_serials[0])

    rgb_resolution = (cfgs.width, cfgs.height)
    depth_resolution = (cfgs.width, cfgs.height)
    camera = RGBDCamera(device_serials[0], rgb_resolution, depth_resolution)
    camera.start()
    rgb_intrinsics, _, _, _ = camera.get_intrinsics_raw()
    fx, fy = rgb_intrinsics.fx, rgb_intrinsics.fy
    cx, cy = rgb_intrinsics.ppx, rgb_intrinsics.ppy
    depth_scale = camera.get_depth_scale()
    print("depth_scale:", depth_scale)

    # === Robot Init ===
    fr3_robot = FrankaController(robot_ip="192.168.1.1")
    print("Move robot to home...")
    fr3_robot.move_home()
    fr3_robot.open_gripper()

    # === GraspNet Init ===
    graspnet_infer = GraspNetInfer(cfgs)

    # warmup
    for _ in range(5):
        camera.shoot()

    workspace_mask = create_rect_mask((cfgs.height, cfgs.width), (200, 0), (1150, 600))

    print("Start real-time grasp prediction ...")
    try:
        while True:
            # === Step 1: 获取RGB-D帧 ===
            rgb_image, depth_image = camera.shoot()
            rgb_image = rgb_image / 255.0
            depth_image = depth_image * depth_scale
            depth_image[depth_image > 1.2] = 0

            # === Step 2: GraspNet预测 ===
            end_points, cloud = graspnet_infer.process_realsense_data(
                rgb_image, depth_image, fx, fy, cx, cy, workspace_mask
            )

            target_gg = graspnet_infer.predict_grasps(end_points, cloud)

            grippers = target_gg.to_open3d_geometry_list()
            T = np.diag([1, 1, -1, 1])
            cloud.transform(T)
            for g in grippers:
                g.transform(T)
            o3d.visualization.draw_geometries([cloud, *grippers])

            target_pose_base = fr3_robot.compute_target_pose(target_gg[0])

            fr3_robot.execute_grasp(target_pose_base)

    except KeyboardInterrupt:
        print("Exit demo.")
        camera.stop()


if __name__ == "__main__":
    main()
