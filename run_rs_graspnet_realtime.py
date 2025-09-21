import os
import sys
import numpy as np
import open3d as o3d
import torch
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


def process_realsense_data(rgb_image, depth_image, rgb_intrinsics, workspace_mask=None):
    """
    将 Realsense 实时 RGB-D 图像转换成 GraspNet 输入
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

    cloud = create_point_cloud_from_depth_image(depth_image, camera_info, organized=True)
    if workspace_mask is None:
        mask = depth_image > 0
    else:
        mask = (depth_image > 0) & (workspace_mask > 0)

    cloud_masked = cloud[mask]
    color_masked = rgb_image[mask]

    # sample points
    num_point = cfgs.num_point
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # Open3D 点云
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # 转 torch
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

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]

    grippers = gg.to_open3d_geometry_list()

    T = np.diag([1, -1, -1, 1])
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

    net = get_net()

    # 丢掉前几帧让相机稳定
    for _ in range(5):
        camera.shoot()

    workspace_mask = create_rect_mask((720, 1280), (200, 0), (1150, 600))
        
    print("Start real-time grasp prediction...")
    try:
        while True:
            rgb_image, depth_image = camera.shoot()
            rgb_image = rgb_image / 255.0  # RGB

            print("rgb: ", rgb_image.shape)
            print("rgb max: ", rgb_image.max())

            depth_image = depth_image * depth_scale
            depth_image[depth_image > 1.2] = 0
            print("depth max: ", depth_image.max())

            # workspace_mask = None
            end_points, cloud = process_realsense_data(rgb_image, depth_image, rgb_intrinsics, workspace_mask)
            gg = get_grasps(net, end_points)
            if cfgs.collision_thresh > 0:
                gg = collision_detection(gg, cloud)
            vis_grasps(gg, cloud)

    except KeyboardInterrupt:
        print("Exit demo.")
        camera.stop()


if __name__ == "__main__":
    demo_realsense()
