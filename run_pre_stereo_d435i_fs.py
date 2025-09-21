import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ---------------------------
# 1. 配置 RealSense 流
# ---------------------------
pipeline = rs.pipeline()
config = rs.config()

# 启用左右红外流（IR），注意这里采集的原始数据为单通道（灰度）
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)  # 左IR
config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)  # 右IR

# 启用彩色流（RGB）
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# 启用深度流（后续可用来对齐彩色）
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 启动设备
profile = pipeline.start(config)

# ---------------------------
# 2. 获取内参并生成 K.txt
# ---------------------------
# 获取左IR内参
left_profile = profile.get_stream(rs.stream.infrared, 1)
left_intrin = left_profile.as_video_stream_profile().get_intrinsics()
K_left = np.array([[left_intrin.fx, 0, left_intrin.ppx],
                   [0, left_intrin.fy, left_intrin.ppy],
                   [0, 0, 1]])
dist_left = np.array(left_intrin.coeffs)

# 基线距离（D435i左右IR间距，单位：米）
baseline = 0.05

# ---------------------------
# 5. 创建保存目录
# ---------------------------
save_dir = r"./log/stereo_d435i"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 保存内参到 K.txt（第一行为展平的3×3矩阵，第二行为基线）
with open(os.path.join(save_dir, 'K.txt'), 'w') as f:
    f.write(' '.join(map(str, K_left.flatten())) + '\n')
    f.write(str(baseline) + '\n')
print("内参矩阵 K 已保存到 K.txt")

# ---------------------------
# 3. 生成左右IR图像的校正映射表
# ---------------------------
# 左IR校正映射
map_left_x, map_left_y = cv2.initUndistortRectifyMap(
    K_left, dist_left, None, K_left, (left_intrin.width, left_intrin.height), cv2.CV_32FC1)

# 对右IR，获取内参（如果需要分别校正，可采用右IR内参）
right_profile = profile.get_stream(rs.stream.infrared, 2)
right_intrin = right_profile.as_video_stream_profile().get_intrinsics()
K_right = np.array([[right_intrin.fx, 0, right_intrin.ppx],
                    [0, right_intrin.fy, right_intrin.ppy],
                    [0, 0, 1]])
dist_right = np.array(right_intrin.coeffs)
map_right_x, map_right_y = cv2.initUndistortRectifyMap(
    K_right, dist_right, None, K_right, (right_intrin.width, right_intrin.height), cv2.CV_32FC1)

# ---------------------------
# 4. 创建对齐对象：将深度图对齐到彩色图（后续可用于上色点云）
# ---------------------------
align_to = rs.stream.color
align = rs.align(align_to)


# ---------------------------
# 6. 循环采集、处理并显示图像
# ---------------------------
try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()
        # 对齐深度帧到彩色帧（用于后续点云上色）
        aligned_frames = align.process(frames)

        # 分别获取各个帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        left_ir_frame = frames.get_infrared_frame(1)
        right_ir_frame = frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not left_ir_frame or not right_ir_frame:
            continue

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        left_ir_image = np.asanyarray(left_ir_frame.get_data())
        right_ir_image = np.asanyarray(right_ir_frame.get_data())

        # 对IR图像进行校正（去畸变）
        rect_left = cv2.remap(left_ir_image, map_left_x, map_left_y, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_ir_image, map_right_x, map_right_y, cv2.INTER_LINEAR)

        # 注意：FoundationStereo 要求输入左右视图为RGB格式（3通道），而当前IR图像为灰度
        # 故这里将校正后的IR图像转换为伪RGB（简单复制灰度通道）
        rect_left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
        rect_right_rgb = cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR)

        # 同时将深度图转换为伪彩色，便于显示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 显示窗口
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Image (Colormap)', depth_colormap)
        cv2.imshow('Left IR (Rectified as RGB)', rect_left_rgb)
        cv2.imshow('Right IR (Rectified as RGB)', rect_right_rgb)

        # 按下 'q' 键退出，并保存图像
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            # 保存校正后的左右IR图像（伪RGB格式，符合 FoundationStereo 要求）
            cv2.imwrite(os.path.join(save_dir, "rect_left.png"), rect_left_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(os.path.join(save_dir, "rect_right.png"), rect_right_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # 保存彩色图和深度伪彩色图（可用于后续点云上色）
            cv2.imwrite(os.path.join(save_dir, "color_image2.png"), color_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(os.path.join(save_dir, "depth_image.png"), depth_colormap, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print("图像已保存到文件夹:", save_dir)
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()