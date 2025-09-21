#!/usr/bin/env python3
"""
fs_graspnet_realtime.py

实时从 D435i 读取左右 IR + RGB，使用 FoundationStereo 预测深度，并将处理后的点云输入 GraspNet 做抓取推理与碰撞检测。

用法示例：
python fs_graspnet_realtime.py --ckpt_fs ./checkpoints/foundation_stereo/11-33-40/model_best_bp2.pth \
    --checkpoint_grasp ./checkpoints/graspnet/checkpoint-rs.tar \
    --num_point 20000 --voxel_size 0.003 --scale 0.5
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import cv2
import torch
import open3d as o3d
import pyrealsense2 as rs
from omegaconf import OmegaConf

# FoundationStereo imports (assume in PYTHONPATH)
from foundation_stereo.core.utils.utils import InputPadder
from foundation_stereo.core.foundation_stereo import FoundationStereo
from foundation_stereo.Utils import set_logging_format, set_seed, depth2xyzmap

# GraspNet imports (assume in PYTHONPATH)
from graspnetAPI import GraspGroup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from collision_detector import ModelFreeCollisionDetector
# your project's modules
from graspnet import GraspNet, pred_decode
from franka_graspnet.realsense import RGBDCamera, get_devices
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_fs', type=str, default="checkpoints/foundation_stereo/11-33-40/model_best_bp2.pth", help='FoundationStereo checkpoint (.pth)')
parser.add_argument('--ckpt_grasp', type=str, default='checkpoints/graspnet/checkpoint-rs.tar', help='GraspNet checkpoint (.tar)')
parser.add_argument('--device', type=str, default='cuda', help='device for models')
parser.add_argument('--scale', type=float, default=0.5, help='downscale factor for IR images used by FS (<=1.0)')
parser.add_argument('--inference_skip', type=int, default=1, help='run FS every N frames')
parser.add_argument('--num_point', type=int, default=20000, help='number of points for GraspNet input')
parser.add_argument('--voxel_size', type=float, default=0.003, help='voxel size for downsampling (meters)')
parser.add_argument('--workspace_z_min', type=float, default=0.01, help='workspace z min (meters)')
parser.add_argument('--workspace_z_max', type=float, default=1.2, help='workspace z max (meters)')
parser.add_argument('--baseline', type=float, default=0.05, help='stereo baseline (m)')
parser.add_argument('--valid_iters', type=int, default=32, help='FoundationStereo valid_iters')
parser.add_argument('--z_far', type=float, default=10.0, help='max depth to keep (m)')
parser.add_argument('--realsense_width', type=int, default=640)
parser.add_argument('--realsense_height', type=int, default=480)
parser.add_argument('--realsense_fps', type=int, default=30)
parser.add_argument('--save_dir', type=str, default='./log/fs_graspnet', help='save dir for K.txt, debug')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
set_logging_format()
set_seed(0)
torch.autograd.set_grad_enabled(False)

# ---------------------------
# Utility: load GraspNet model
# ---------------------------
def get_graspnet_model(checkpoint_path):
    cfg_local = argparse.Namespace()  # mimic cfgs in your code
    cfg_local.num_view = 300
    cfg_local.num_point = args.num_point
    # create net (parameters from user snippet)
    net = GraspNet(input_feature_dim=0, num_view=cfg_local.num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    net.to(device)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print(f"Loaded GraspNet from {checkpoint_path} -> device {device}")
    return net, device

# decode + wrapper (from user snippet)
def get_grasps_from_net(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    return GraspGroup(gg_array)

# collision detection wrapper
def collision_detection(gg: GraspGroup, cloud_o3d: o3d.geometry.PointCloud, voxel_size=0.01, collision_thresh=0.01):
    mfcdet = ModelFreeCollisionDetector(np.asarray(cloud_o3d.points), voxel_size=voxel_size)
    collision_mask = mfcdet.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg

# visualization wrapper (display top grasps)
def vis_grasps(gg: GraspGroup, cloud_o3d: o3d.geometry.PointCloud, topk=50):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:topk]
    grippers = gg.to_open3d_geometry_list()
    # transform for visualization (match your earlier T)
    T = np.diag([1, -1, -1, 1])
    cloud_vis = cloud_o3d.translate((0,0,0), relative=False)  # copy? open3d uses references; we'll transform local copy
    cloud_copy = o3d.geometry.PointCloud(cloud_vis)
    cloud_copy.transform(T)
    for g in grippers:
        g.transform(T)
    o3d.visualization.draw_geometries([cloud_copy, *grippers])

# ---------------------------
# Initialize RealSense pipeline (for IR1, IR2, color, depth)
# ---------------------------
def init_realsense(width, height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    return pipeline, profile, config

# ---------------------------
# Load FoundationStereo model
# ---------------------------
def load_foundation_stereo(ckpt_path):
    ckpt_dir = ckpt_path
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    # overwrite with runtime args
    for k, v in vars(args).items():
        cfg[k] = v
    cfg = OmegaConf.create(cfg)
    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(args.device)
    model.eval()
    print(f"Loaded FoundationStereo from {ckpt_path} on {args.device}")
    return model, cfg

# ---------------------------
# Pointcloud preprocessing pipeline
# ---------------------------
def preprocess_pointcloud(pts: np.ndarray, colors: np.ndarray,
                          voxel_size=0.003, num_point=20000,
                          z_min=0.01, z_max=1.2):
    """
    pts: (N,3) in meters, colors: (N,3) in [0,1]
    Returns: cloud_o3d, pts_sampled (num_point,3), colors_sampled (num_point,3)
    """
    # 1) filter NaN/inf
    mask_finite = np.isfinite(pts).all(axis=1)
    pts = pts[mask_finite]
    colors = colors[mask_finite]

    # 2) workspace z clip
    mask_ws = (pts[:,2] > z_min) & (pts[:,2] < z_max)
    pts = pts[mask_ws]
    colors = colors[mask_ws]

    # 3) drop too close/zero points
    keep = np.linalg.norm(pts, axis=1) > 1e-6
    pts = pts[keep]
    colors = colors[keep]

    # If empty
    if pts.shape[0] == 0:
        return None, None, None

    # 4) Open3D pointcloud + voxel downsample
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size=voxel_size)

    pts_ds = np.asarray(cloud.points)
    colors_ds = np.asarray(cloud.colors)

    if pts_ds.shape[0] == 0:
        return None, None, None

    # 5) random sample to fixed num_point
    if pts_ds.shape[0] >= num_point:
        idxs = np.random.choice(pts_ds.shape[0], num_point, replace=False)
    else:
        idxs = np.random.choice(pts_ds.shape[0], num_point, replace=True)
    pts_sampled = pts_ds[idxs]
    colors_sampled = colors_ds[idxs]

    return cloud, pts_sampled, colors_sampled

# ---------------------------
# FS inference helper: get depth map (m) and point cloud (pts_keep, colors_keep)
# ---------------------------
def fs_infer_and_pointcloud(model, left_rgb, right_rgb, color_image, K_left, baseline, scale, valid_iters, z_far):
    """
    left_rgb/right_rgb: HxWx3 (uint8 or float). color_image: HxWx3 RGB BGR? expects BGR (from camera)
    K_left: original intrinsics 3x3
    scale: scale applied to images (0<scale<=1)
    Returns:
      depth_m (H,W) in meters (with inf for invalid), pts_keep (M,3), colors_keep (M,3)
    """
    # prepare tensors
    img0 = torch.as_tensor(left_rgb).to(args.device).float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(right_rgb).to(args.device).float()[None].permute(0,3,1,2)

    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0_pad, img1_pad = padder.pad(img0, img1)

    with torch.amp.autocast(device_type=("cuda" if "cuda" in args.device else "cpu")):
        disp = model.forward(img0_pad, img1_pad, iters=valid_iters, test_mode=True)
    disp = padder.unpad(disp.float())
    H = img0.shape[2]; W = img0.shape[3]
    disp_np = disp.data.cpu().numpy().reshape(H, W)

    # invalid
    disp_np[disp_np <= 0] = np.inf

    # depth conversion (meters)
    fx = K_left[0,0] * scale
    depth_m = (fx * baseline) / disp_np
    depth_m[~np.isfinite(depth_m)] = np.inf

    # convert depth->xyz map
    K_scaled = K_left.copy().astype(np.float32)
    K_scaled[:2] *= scale
    xyz_map = depth2xyzmap(depth_m, K_scaled)  # HxWx3

    # flatten and filter valid
    pts = xyz_map.reshape(-1,3)
    # color_image assumed BGR from camera; convert to RGB normalized [0,1]
    color_small = color_image
    if color_small.dtype != np.float32:
        color_small = color_small.astype(np.float32)
    if color_small.max() > 1.1:
        color_small = color_small / 255.0
    colors = cv2.cvtColor(color_small, cv2.COLOR_BGR2RGB).reshape(-1,3)

    valid_mask = np.isfinite(pts).all(axis=1) & (pts[:,2] > 0) & (pts[:,2] <= z_far)
    pts_keep = pts[valid_mask]
    colors_keep = colors[valid_mask]
    return depth_m, pts_keep, colors_keep

# ---------------------------
# Main loop
# ---------------------------
def main():
    # init models
    model_fs, cfg_fs = load_foundation_stereo(args.ckpt_fs)
    net_grasp, device_grasp = get_graspnet_model(args.ckpt_grasp)

    # start realsense
    width = args.realsense_width
    height = args.realsense_height
    fps = args.realsense_fps
    pipeline, profile, config = None, None, None
    try:
        pipeline, profile, config = init_realsense(width, height, fps)
    except Exception as e:
        print("Failed to start RealSense pipeline:", e)
        return

    # get intrinsics (left IR)
    left_profile = profile.get_stream(rs.stream.infrared, 1)
    left_intrin = left_profile.as_video_stream_profile().get_intrinsics()
    K_left = np.array([[left_intrin.fx, 0, left_intrin.ppx],
                       [0, left_intrin.fy, left_intrin.ppy],
                       [0, 0, 1]], dtype=np.float32)

    # prepare undistort maps for IR -> rectified
    right_profile = profile.get_stream(rs.stream.infrared, 2)
    right_intrin = right_profile.as_video_stream_profile().get_intrinsics()
    K_right = np.array([[right_intrin.fx, 0, right_intrin.ppx],
                        [0, right_intrin.fy, right_intrin.ppy],
                        [0, 0, 1]], dtype=np.float32)
    dist_left = np.array(left_intrin.coeffs, dtype=np.float32)
    dist_right = np.array(right_intrin.coeffs, dtype=np.float32)
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(K_left, dist_left, None, K_left, (width, height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(K_right, dist_right, None, K_right, (width, height), cv2.CV_32FC1)

    # align depth to color so we can sample color for FS output
    align = rs.align(rs.stream.color)

    # save K.txt
    with open(os.path.join(args.save_dir, 'K.txt'), 'w') as f:
        f.write(' '.join(map(str, K_left.flatten())) + '\n')
        f.write(str(args.baseline) + '\n')
    print("Saved intrinsic to", os.path.join(args.save_dir, 'K.txt'))

    # open3d visualization object for live point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FS -> PointCloud (live)", width=960, height=720)
    pcd_vis = o3d.geometry.PointCloud()
    added = False

    frame_idx = 0
    try:
        print("Start main loop. Press Ctrl+C to exit.")
        while True:
            # robust wait_for_frames with restart on timeout
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print("Frame timeout/Error:", e, " -> restarting pipeline")
                try:
                    pipeline.stop()
                except:
                    pass
                time.sleep(0.5)
                pipeline, profile, config = init_realsense(width, height, fps)
                # refresh intrinsics/maps if needed (skipped for brevity)
                continue

            # align depth to color for color sampling
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            left_ir_frame = frames.get_infrared_frame(1)
            right_ir_frame = frames.get_infrared_frame(2)

            if not color_frame or not depth_frame or not left_ir_frame or not right_ir_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())  # BGR
            depth_raw = np.asanyarray(depth_frame.get_data())
            left_ir = np.asanyarray(left_ir_frame.get_data())
            right_ir = np.asanyarray(right_ir_frame.get_data())

            # undistort / rectify IRs
            rect_left = cv2.remap(left_ir, map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
            rect_right = cv2.remap(right_ir, map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)

            # convert to 3-ch pseudo RGB
            left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
            right_rgb = cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR)

            # resize according to scale
            scale = float(args.scale)
            if scale != 1.0:
                H0, W0 = left_rgb.shape[:2]
                newW, newH = int(W0 * scale), int(H0 * scale)
                left_in = cv2.resize(left_rgb, (newW, newH), interpolation=cv2.INTER_LINEAR)
                right_in = cv2.resize(right_rgb, (newW, newH), interpolation=cv2.INTER_LINEAR)
                color_vis = cv2.resize(color_image, (newW, newH), interpolation=cv2.INTER_LINEAR)
            else:
                left_in = left_rgb
                right_in = right_rgb
                color_vis = color_image.copy()

            # show debug windows (optional)
            cv2.imshow('Color', color_image)
            cv2.imshow('Left IR (rect)', left_in)
            cv2.imshow('Right IR (rect)', right_in)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # optionally skip FS inference to save GPU
            if frame_idx % max(1, args.inference_skip) != 0:
                frame_idx += 1
                continue

            # FS inference -> depth and pointcloud
            depth_m, pts_keep, colors_keep = fs_infer_and_pointcloud(
                model_fs, left_in, right_in, color_vis, K_left, args.baseline, scale, args.valid_iters, args.z_far
            )
            if pts_keep is None or pts_keep.shape[0] == 0:
                print("No valid FS points, skipping frame.")
                frame_idx += 1
                continue

            # Preprocess pointcloud for GraspNet
            cloud_o3d, pts_sampled, colors_sampled = preprocess_pointcloud(
                pts_keep, colors_keep, voxel_size=args.voxel_size, num_point=args.num_point,
                z_min=args.workspace_z_min, z_max=args.workspace_z_max
            )
            if cloud_o3d is None:
                print("Pointcloud empty after preprocessing, skipping.")
                frame_idx += 1
                continue

            # Build GraspNet input (point_clouds tensor and colors)
            pts_tensor = torch.from_numpy(pts_sampled[np.newaxis].astype(np.float32)).to(device_grasp)
            # some GraspNet implementations accept colors in end_points; include as numpy if needed
            end_points = {'point_clouds': pts_tensor, 'cloud_colors': colors_sampled}

            # Grasp inference
            gg = get_grasps_from_net(net_grasp, end_points)

            # collision detection
            gg = collision_detection(gg, cloud_o3d, voxel_size=0.01, collision_thresh=0.01)

            # visualize grasps + cloud (blocking)
            vis_grasps(gg, cloud_o3d)

            # also update live FS point cloud viewer (non-blocking)
            # update pcd_vis to show the dense (downsampled) cloud
            pcd_vis.points = cloud_o3d.points
            pcd_vis.colors = cloud_o3d.colors
            if not added:
                vis.add_geometry(pcd_vis)
                added = True
            else:
                vis.update_geometry(pcd_vis)
            vis.poll_events()
            vis.update_renderer()

            frame_idx += 1

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        try:
            pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()
        try:
            vis.destroy_window()
        except:
            pass
        print("Exiting.")

if __name__ == "__main__":
    main()
