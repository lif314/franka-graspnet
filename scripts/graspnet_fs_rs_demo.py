import argparse
import numpy as np
import open3d as o3d
import sys
import traceback
import cv2
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..'))
from franka_graspnet.graspnet_infer import GraspNetInfer
from franka_graspnet.stereo_realsense import StereoCameraIR
from foundation_stereo.fs_infer import FoundationStereoInfer

def create_rect_mask(image_shape, top_left, bottom_right):
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

    # FS
    parser.add_argument('--fs_ckpt_dir', type=str, default='./checkpoints/foundation_stereo/11-33-40/model_best_bp2.pth')
    parser.add_argument('--baseline', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--hiera', default=0, type=int)
    parser.add_argument('--z_far', default=10, type=float)
    parser.add_argument('--valid_iters', type=int, default=32)
    parser.add_argument('--denoise_cloud', type=int, default=1)
    parser.add_argument('--denoise_nb_points', type=int, default=30)
    parser.add_argument('--denoise_radius', type=float, default=0.03)
    args = parser.parse_args()

    # === Camera Init ===
    print("Init realsense camera.......")
    rs_dev = StereoCameraIR(width=args.width, height=args.height, fps=30)
    try:
        rs_dev.start()
    except Exception as e:
        print("Failed to start RealSense:", e)
        traceback.print_exc()
        sys.exit(1)
    
    # IR to Color
    R_ir2color, T_ir2color = rs_dev.get_ir2color()
    # Color Intrinsics
    fx_c, fy_c, ppx_c, ppy_c = rs_dev.get_color_intrinsics()

    # IR to Color
    R_ir2color, T_ir2color = rs_dev.get_ir2color()
    # Color Intrinsics
    fx_c, fy_c, ppx_c, ppy_c = rs_dev.get_color_intrinsics()

    fs = FoundationStereoInfer(ckpt_dir=args.fs_ckpt_dir, device=args.device, baseline=args.baseline,
            valid_iters=args.valid_iters, hiera=args.hiera)
    
    # === GraspNet Init ===
    graspnet_infer = GraspNetInfer(args)

    workspace_mask = create_rect_mask((args.height, args.width), (200, 0), (1150, 600))

    print("Start real-time grasp prediction ...")
    try:
        while True:
            res = rs_dev.get_frames()
            if res is None:
                continue
            color_image, depth_raw, left_rgb, right_rgb, K_left = res

            scale = float(args.scale)
            H0,W0 = left_rgb.shape[:2]
            newW,newH = int(W0*scale), int(H0*scale)
            if scale != 1.0:
                left_rgb = cv2.resize(left_rgb,(newW,newH))
                right_rgb = cv2.resize(right_rgb,(newW,newH))
                color_vis = cv2.resize(color_image,(newW,newH))
            else:
                color_vis = color_image.copy()

            depth_m, valid_mask, K_scaled = fs.infer_depth(left_rgb, right_rgb, K_left, scale=scale)
            depth_m[np.isinf(depth_m)] = 0
            depth_m[depth_m > 1.2] = 0
            valid_mask = valid_mask & (depth_m > 0) & (depth_m <= args.z_far) & (workspace_mask > 0)
            if np.any(valid_mask):
                print(f"Depth min/max (valid only): {depth_m[valid_mask].min():.3f}/{depth_m[valid_mask].max():.3f}")

            H, W = depth_m.shape
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            z = depth_m

            fx = K_scaled[0,0]; ppx = K_scaled[0,2]
            fy = K_scaled[1,1]; ppy = K_scaled[1,2]
            x = (xx - ppx) / fx * z
            y = (yy - ppy) / fy * z
            points_ir = np.stack([x, y, z], axis=-1).reshape(-1,3)
            mask_flat = valid_mask.reshape(-1)

            # transform LEFT IR points to Color
            points_color = (R_ir2color @ points_ir.T).T + T_ir2color

            # project to color image
            pts_z = points_color[:,2]
            valid_z = pts_z > 1e-6
            u = (points_color[:,0] / np.where(valid_z, pts_z, 1.0)) * fx_c + ppx_c
            v = (points_color[:,1] / np.where(valid_z, pts_z, 1.0)) * fy_c + ppy_c
            u_int = np.round(u).astype(np.int32)
            v_int = np.round(v).astype(np.int32)

            in_bounds = (u_int >= 0) & (u_int < color_vis.shape[1]) & (v_int >= 0) & (v_int < color_vis.shape[0]) & valid_z
            final_mask = mask_flat & in_bounds
            idxs = np.where(final_mask)[0]

            pts_keep = points_color[idxs]
            u_sel = u_int[idxs]; v_sel = v_int[idxs]
            colors_keep = cv2.cvtColor(color_vis, cv2.COLOR_BGR2RGB)[v_sel, u_sel, :].astype(np.float32) / 255.0

            end_points, cloud = graspnet_infer.process_fs_data(pts_keep, colors_keep)

            target_gg = graspnet_infer.predict_grasps(end_points, cloud, return_best=False)

            grippers = target_gg.to_open3d_geometry_list()
            T = np.diag([1, 1, -1, 1])
            cloud.transform(T)
            for g in grippers:
                g.transform(T)
            o3d.visualization.draw_geometries([cloud, *grippers])

    except KeyboardInterrupt:
        print("Exit demo.")
        rs_dev.stop()


if __name__ == "__main__":
    main()
