import pyrealsense2 as rs
import numpy as np
import cv2

class StereoCameraIR:
    """RealSense helper: start/stop pipeline, rectification maps, frame retrieval."""

    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
        self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = None
        self.ir_profile = None
        self.color_profile = None
        self.map_left_x = self.map_left_y = None
        self.map_right_x = self.map_right_y = None
        self.K_left = None
        self.dist_left = None
        self.K_right = None
        self.dist_right = None
        self.align = None

    def start(self):
        self.profile = self.pipeline.start(self.config) 
        self.ir_profile = self.profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        self.color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()

        # left IR intrinsics
        left_profile = self.profile.get_stream(rs.stream.infrared, 1)
        left_intrin = left_profile.as_video_stream_profile().get_intrinsics()
        self.K_left = np.array([[left_intrin.fx, 0, left_intrin.ppx],
                                [0, left_intrin.fy, left_intrin.ppy],
                                [0, 0, 1]], dtype=np.float32)
        self.dist_left = np.array(left_intrin.coeffs, dtype=np.float32)

        # right IR intrinsics
        right_profile = self.profile.get_stream(rs.stream.infrared, 2)
        right_intrin = right_profile.as_video_stream_profile().get_intrinsics()
        self.K_right = np.array([[right_intrin.fx, 0, right_intrin.ppx],
                                 [0, right_intrin.fy, right_intrin.ppy],
                                 [0, 0, 1]], dtype=np.float32)
        self.dist_right = np.array(right_intrin.coeffs, dtype=np.float32)

        # undistort maps
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.K_left, self.dist_left, None, self.K_left, (self.width, self.height), cv2.CV_32FC1)
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.K_right, self.dist_right, None, self.K_right, (self.width, self.height), cv2.CV_32FC1)

        # align depth to color
        self.align = rs.align(rs.stream.color)

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


    def get_ir2color(self):
        # IR -> COLOR extrinsics
        extrin_ir2color = self.ir_profile.get_extrinsics_to(self.color_profile)
        R_ir2color = np.array(extrin_ir2color.rotation).reshape(3,3)
        T_ir2color = np.array(extrin_ir2color.translation).reshape(3)
        return R_ir2color, T_ir2color
    
    def get_color_intrinsics(self):
        color_intrin = self.color_profile.get_intrinsics()
        fx_c, fy_c = color_intrin.fx, color_intrin.fy
        ppx_c, ppy_c = color_intrin.ppx, color_intrin.ppy
        return fx_c, fy_c, ppx_c, ppy_c

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        left_ir_frame = frames.get_infrared_frame(1)
        right_ir_frame = frames.get_infrared_frame(2)

        if not color_frame or not depth_frame or not left_ir_frame or not right_ir_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        left_ir = np.asanyarray(left_ir_frame.get_data())
        right_ir = np.asanyarray(right_ir_frame.get_data())

        rect_left = cv2.remap(left_ir, self.map_left_x, self.map_left_y, interpolation=cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_ir, self.map_right_x, self.map_right_y, interpolation=cv2.INTER_LINEAR)

        left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR)
        right_rgb = cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR)

        return color_image, depth_raw, left_rgb, right_rgb, self.K_left.copy()