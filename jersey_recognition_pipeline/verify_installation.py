import torch
from mmcv import __version__ as mmcv_version
from mmdet import __version__ as mmdet_version
from mmpose import __version__ as mmpose_version

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"MMCV: {mmcv_version}")
print(f"MMDet: {mmdet_version}")
print(f"MMPose: {mmpose_version}")
