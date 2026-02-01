from re import T
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import timm
import time

from navsim.common.enums import StateSE2Index

import torchvision.models as models
import torch.nn.functional as F
from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs

import os
from datetime import datetime
from navsim.agents.ImagineWorld.ImagineWorld_targets import BoundingBox2DIndex
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from typing import Any, List, Dict, Union
from navsim.agents.ImagineWorld.ImagineWorld_backbone import ImagineWorldBackbone

class ExpertImagineWorldModel(nn.Module):
    def __init__(self, 
                 config,
                ):
        super().__init__()

        # Define constants as variables
        STATUS_ENCODING_INPUT_DIM = 4 + 2 + 2
        hidden_dim = 256
        NUM_CLUSTERS = config.n_clusters if hasattr(config, 'n_clusters') else 256
        CLUSTER_CENTERS_FEATURE_DIM = 24

        TRANSFORMER_DIM_FEEDFORWARD = 512
        TRANSFORMER_NHEAD = 8
        TRANSFORMER_DROPOUT = 0.1
        TRANSFORMER_NUM_LAYERS = 2

        SCORE_HEAD_HIDDEN_DIM = 128
        SCORE_HEAD_OUTPUT_DIM = 1
        NUM_SCORE_HEADS = 5

        # transfuser backbone
        self._backbone = TransfuserBackbone(config)
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._checkpoint_path = config.expert_checkpoint_path
        self.initialize()
    
    def initialize(self) -> None:
        """Inherited, see superclass."""
        #import pdb;pdb.set_trace()
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)#["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )["state_dict"]
        
        if "agent.ImagineWorld_model.trajectory_anchors" in state_dict:
            del state_dict["agent.ImagineWorld_model.trajectory_anchors"]

        self.load_state_dict({k.replace("agent.expert_ImagineWorld_model.", ""): v for k, v in state_dict.items()}, strict=False)
        #import pdb;pdb.set_trace()
        #print()

    def forward(self,features,targets):

        camera_feature = features["camera_feature"]
        lidar_feature = features["lidar_feature"]
        _, backbone_bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        import pdb;pdb.set_trace()
        bev_feature = self._bev_downscale(backbone_bev_feature).flatten(-2, -1).permute(0, 2, 1)
        # for name, param in self._backbone.named_parameters():
        #     print(name, param)
        #     import pdb;pdb.set_trace()
        #     print()
        #import pdb;pdb.set_trace()
        return bev_feature