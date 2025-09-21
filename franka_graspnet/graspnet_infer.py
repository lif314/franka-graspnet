import os
import sys
import torch
import numpy as np
import open3d as o3d

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..', 'models'))
sys.path.append(os.path.join(ROOT_DIR, '..', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

class GraspNetInfer:
    def __init__(self, cfgs):
        self.checkpoint_path = cfgs.checkpoint_path
        self.num_view = cfgs.num_view
        self.num_point = cfgs.num_point
        self.voxel_size = cfgs.voxel_size
        self.collision_thresh = cfgs.collision_thresh
        self.angle_threshold_deg = cfgs.angle_threshold_deg
        self.top_k_grasps = 20

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.get_net()

    def get_net(self):
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                    cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        print("-> loaded checkpoint", self.checkpoint_path)
        return net

    def process_realsense_data(self, rgb_image, depth_image, fx, fy, cx, cy, workspace_mask=None,
                           plane_remove=True, plane_dist_thresh=0.2):
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
        num_point = self.num_point
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
    
    def process_fs_data(self, points, pts_colors):
        
        # 3. 采样
        num_point = self.num_point
        if len(points) >= num_point:
            idxs = np.random.choice(len(points), num_point, replace=False)
        else:
            idxs1 = np.arange(len(points))
            idxs2 = np.random.choice(len(points), num_point - len(points), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = points[idxs]
        color_sampled = pts_colors[idxs]

        # 4. Open3D 点云 (可视化用)
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(points.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(pts_colors.astype(np.float32))

        # 5. 转 torch (GraspNet 输入)
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)

        end_points = dict()
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled
        return end_points, cloud_o3d

    def predict_grasps(self, end_points, cloud):
        """
        预测抓取姿态
        
        Args:
            end_points: 点云数据字典
            cloud: Open3D点云对象
            visual: 是否可视化结果
            
        Returns:
            GraspGroup: 最佳抓取预测结果
        """
        # 1. 前向推理
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        
        # 2. 碰撞检测
        if self.collision_thresh > 0:
            gg = self._collision_detection(gg, cloud)
        
        # 3. NMS去重 + 按置信度排序
        gg.nms().sort_by_score()
        
        # 4. 垂直角度筛选
        filtered_grasps = self._filter_by_vertical_angle(gg)
        
        # 5. 选择最佳抓取
        best_grasp_group = self._select_best_grasp(filtered_grasps)
        
        return best_grasp_group
    
    def _collision_detection(self, grasp_group, cloud):
        """执行碰撞检测"""
        mfcdetector = ModelFreeCollisionDetector(
            np.asarray(cloud.points), 
            voxel_size=self.voxel_size
        )
        collision_mask = mfcdetector.detect(
            grasp_group, 
            approach_dist=0.05, 
            collision_thresh=self.collision_thresh
        )
        return grasp_group[~collision_mask]
    
    def _filter_by_vertical_angle(self, grasp_group):
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
            
            if angle < np.deg2rad(self.angle_threshold_deg):
                filtered.append(grasp)
        
        if len(filtered) == 0:
            print(f"\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
            filtered = all_grasps
        else:
            print(f"\nFiltered {len(filtered)} grasps within ±{self.angle_threshold_deg}° of vertical out of {len(all_grasps)} total predictions.")
        
        return filtered
    
    def _select_best_grasp(self, filtered_grasps):
        """选择最佳抓取并返回GraspGroup"""
        # 按得分排序
        filtered_grasps.sort(key=lambda g: g.score, reverse=True)
        
        # 取前k个抓取
        top_grasps = filtered_grasps[:min(len(filtered_grasps), self.top_k_grasps)]
        
        # 选择得分最高的抓取
        best_grasp = top_grasps[0]
        
        # 创建新的GraspGroup并添加最佳抓取
        result_gg = GraspGroup()
        # for g in top_grasps:
        #     result_gg.add(g)
        result_gg.add(best_grasp)

        return result_gg
    
    def update_parameters(self, **kwargs):
        """更新预测参数"""
        for key, value in kwargs.items():
            if key == 'angle_threshold_deg':
                self.angle_threshold = np.deg2rad(value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter {key}")


    def get_grasps(self, end_points):
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        return GraspGroup(gg_array)

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(np.array(cloud.points), voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg
    
    def vis_grasps(self, gg, cloud, num=50):
        gg.nms()
        gg.sort_by_score()
        gg = gg[:num]

        grippers = gg.to_open3d_geometry_list()

        T = np.diag([1, 1, -1, 1])
        cloud.transform(T)

        for g in grippers:
            g.transform(T)

        o3d.visualization.draw_geometries([cloud, *grippers])
