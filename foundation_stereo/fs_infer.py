
import os
import torch
import numpy as np
import cv2
from omegaconf import OmegaConf
from foundation_stereo.core.utils.utils import InputPadder
from foundation_stereo.core.foundation_stereo import FoundationStereo
from foundation_stereo.Utils import set_seed

class FoundationStereoInfer:
    """FoundationStereo wrapper."""

    def __init__(self, ckpt_dir, device='cuda', baseline=0.05, valid_iters=32, hiera=0):
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.baseline = baseline
        self.valid_iters = valid_iters
        self.hiera = hiera

        cfg_path = os.path.join(os.path.dirname(self.ckpt_dir), 'cfg.yaml')
        cfg = OmegaConf.load(cfg_path)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        for k, v in dict(baseline=self.baseline, valid_iters=self.valid_iters, hiera=self.hiera).items():
            cfg[k] = v
        cfg['device'] = device
        self.args = OmegaConf.create(cfg)
        
        torch.autograd.set_grad_enabled(False)
        print(f"Loading FoundationStereo model from: {self.ckpt_dir}")
        set_seed(0)
        self.model = FoundationStereo(self.args)
        ckpt = torch.load(self.ckpt_dir, weights_only=False, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()

    def infer_depth(self, left_rgb, right_rgb, K_left, scale=1.0):
        if scale != 1.0:
            H0, W0 = left_rgb.shape[:2]
            newW, newH = int(W0 * scale), int(H0 * scale)
            left_in = cv2.resize(left_rgb, (newW, newH), interpolation=cv2.INTER_LINEAR)
            right_in = cv2.resize(right_rgb, (newW, newH), interpolation=cv2.INTER_LINEAR)
        else:
            left_in = left_rgb
            right_in = right_rgb

        img0 = torch.as_tensor(left_in).to(self.device).float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(right_in).to(self.device).float()[None].permute(0,3,1,2)

        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_pad, img1_pad = padder.pad(img0, img1)

        device_type = "cuda" if "cuda" in self.device else "cpu"
        with torch.amp.autocast(device_type=device_type):
            disp = self.model.forward(img0_pad, img1_pad, iters=self.valid_iters, test_mode=True)

        disp = padder.unpad(disp.float())
        H, W = img0.shape[2], img0.shape[3]
        disp_np = disp.data.cpu().numpy().reshape(H,W)
        disp_np[disp_np <= 0] = np.inf

        fx = K_left[0,0]*scale
        depth_m = (fx*self.baseline) / disp_np

        valid_mask = np.isfinite(depth_m) & (depth_m>0)
        K_scaled = K_left.copy().astype(np.float32)
        K_scaled[:2] *= scale

        return depth_m, valid_mask, K_scaled