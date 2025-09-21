#!/usr/bin/env python3
"""
Realtime FS + RealSense with points transformed to color camera coordinate system.
"""

import argparse
import os
import sys
import traceback
import cv2
import numpy as np
import open3d as o3d

from foundation_stereo.fs_infer import FoundationStereoInfer
from franka_graspnet.stereo_realsense import StereoCameraIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/foundation_stereo/11-33-40/model_best_bp2.pth')
    parser.add_argument('--baseline', type=float, default=0.05)
    parser.add_argument('--save_dir', default='./log/fs_rs', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--hiera', default=0, type=int)
    parser.add_argument('--z_far', default=10, type=float)
    parser.add_argument('--valid_iters', type=int, default=32)
    parser.add_argument('--denoise_cloud', type=int, default=1)
    parser.add_argument('--denoise_nb_points', type=int, default=30)
    parser.add_argument('--denoise_radius', type=float, default=0.03)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # RealSense
    rs_dev = StereoCameraIR(width=1280, height=720, fps=30)
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

    fs = FoundationStereoInfer(ckpt_dir=args.ckpt_dir, device=args.device, baseline=args.baseline,
            valid_iters=args.valid_iters, hiera=args.hiera)

    # Open3D window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="FS Realsense PointCloud")
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    render_opt = vis.get_render_option()
    render_opt.point_size = 1.5
    render_opt.background_color = np.array([0.1,0.1,0.1])

    frame_idx = 0
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

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Color Image', color_vis)
            cv2.imshow('Depth (sensor colormap)', depth_colormap)
            cv2.imshow('Left IR', left_rgb)
            cv2.imshow('Right IR', right_rgb)
            cv2.waitKey(1)

            depth_m, valid_mask, K_scaled = fs.infer_depth(left_rgb, right_rgb, K_left, scale=scale)
            depth_m[np.isinf(depth_m)] = 0
            valid_mask = valid_mask & (depth_m > 0) & (depth_m <= args.z_far)
            if np.any(valid_mask):
                print(f"Frame {frame_idx} depth min/max (valid only): {depth_m[valid_mask].min():.3f}/{depth_m[valid_mask].max():.3f}")

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

            if pts_keep.shape[0] > 0:
                pcd.points = o3d.utility.Vector3dVector(pts_keep)
                pcd.colors = o3d.utility.Vector3dVector(colors_keep)
                if args.denoise_cloud and len(pts_keep) >= args.denoise_nb_points:
                    cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
                    pcd = pcd.select_by_index(ind)
                    vis.clear_geometries()
                    vis.add_geometry(pcd)

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            print("按 [空格] 进入下一帧，按 'q' 退出...")
            while True:
                vis.poll_events()
                vis.update_renderer()
                key = cv2.waitKey(50) & 0xFF
                if key == ord(' '):   # 空格：下一帧
                    break
                elif key == ord('q'): # q：退出
                    raise KeyboardInterrupt

            frame_idx += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Exception in main loop:", e)
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        try:
            rs_dev.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("Exited cleanly.")

if __name__ == "__main__":
    main()
