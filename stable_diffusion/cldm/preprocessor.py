import numpy as np

import torch
from einops import rearrange
from torchvision.transforms import Resize, InterpolationMode, CenterCrop, Compose

from cldm.annotator.util import resize_image, HWC3
from cldm.annotator.canny import apply_canny

def detectmap_proc(detected_map, module, rgbbgr_mode, h, w, device):
    detected_map = HWC3(detected_map)
    if module == "normal_map" or rgbbgr_mode:
        control = torch.from_numpy(detected_map[:, :, ::-1].copy()).float().to(device) / 255.0
    else:
        control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
        
    control = rearrange(control, 'h w c -> c h w')
    detected_map = rearrange(torch.from_numpy(detected_map), 'h w c -> c h w')

    h0 = detected_map.shape[1]
    w0 = detected_map.shape[2]
    w1 = w0
    h1 = int(w0/w*h)
    if (h/w > h0/w0):
        h1 = h0
        w1 = int(h0/h*w)
    transform = Compose([
        CenterCrop(size=(h1, w1)),
        Resize(size=(h, w), interpolation=InterpolationMode.BICUBIC)
    ])
    control = transform(control)
    detected_map = transform(detected_map)
    
    # for log use
    detected_map = rearrange(detected_map, 'c h w -> h w c').numpy().astype(np.uint8)
    return control, detected_map

def canny(img, thr_a=50, thr_b=150, **kwargs):
    img = resize_image(HWC3(img), 512)
    result = apply_canny(img, thr_a, thr_b)
    return result