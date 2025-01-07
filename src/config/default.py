from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelConfig:
    checkpoint: str = 'facebook/detr-resnet-50'
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.8

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 4
    num_workers: int = 3
    max_epochs: int = 100
    gradient_clip_val: float = 0.1
    accumulate_grad_batches: int = 8
    log_every_n_steps: int = 5

@dataclass
class DataConfig:
    roboflow_api_key: str = "umCDBYYeGbFwUd9x2KRY"
    workspace: str = "fire-qdubq"
    project: str = "detr_api"
    version: int = 1
    annotation_file_name: str = "_annotations.coco.json"

config = {
    'model': ModelConfig(),
    'training': TrainingConfig(),
    'data': DataConfig()
} 