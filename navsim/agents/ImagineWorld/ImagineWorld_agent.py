from typing import Any, List, Dict, Union
import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from torchvision import transforms

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
import torch.nn as nn
from navsim.common.dataclasses import Scene
import timm, cv2
from navsim.agents.ImagineWorld.ImagineWorld_model import ImagineWorldModel
from navsim.agents.ImagineWorld.Expert_ImagineWorld_model import ExpertImagineWorldModel
from navsim.agents.ImagineWorld.ImagineWorld_loss import compute_ImagineWorld_loss
from navsim.agents.ImagineWorld.ImagineWorld_latent_dit import ImagineWorldLatentDiT
from navsim.agents.ImagineWorld.ImagineWorld_targets import ImagineWorldTargetBuilder
from navsim.agents.ImagineWorld.ImagineWorld_features import ImagineWorldFeatureBuilder
from navsim.agents.ImagineWorld.ImagineWorld_reward_function import ImagineRewardModel
from navsim.agents.ImagineWorld.ImagineWorld_reward_function_simple import ImagineRewardModelSimple
from navsim.agents.ImagineWorld.ImagineWorld_reward_function_seg import ImagineRewardModelSeg
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
import math
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.agents.ImagineWorld.models.DiT import DiT_models

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class ImagineWorldAgent(AbstractAgent):
    def __init__(
        self,
        config,
        trajectory_sampling: TrajectorySampling,
        lr: float,
        checkpoint_path: str = None,
        slice_indices=[3,11],
        resume_from_checkpoint=False,
        use_wm=False,
    ):
        super().__init__()
        self.train_stage = config.train_stage
        self._trajectory_sampling = trajectory_sampling
        self._checkpoint_path = config.checkpoint_path
        self._lr = lr
        self.max_epochs = config.max_epochs if hasattr(config, 'max_epochs') else 100
        self.min_lr = config.min_lr if hasattr(config, 'min_lr') else 1e-6
        if self.train_stage == 'stage1':
            self.ImagineWorld_model = ImagineWorldModel(config)
        elif self.train_stage == 'stage2':
            self.ImagineWorld_model = ImagineWorldModel(config)
            self.latent_world_model = ImagineWorldLatentDiT(config)
        elif self.train_stage == 'stage3':
            self.ImagineWorld_model = ImagineWorldModel(config)
            self.latent_world_model = ImagineWorldLatentDiT(config)
            self.reward_model = ImagineRewardModel(config)
        elif self.train_stage == 'stage4':
            self.ImagineWorld_model = ImagineWorldModel(config)
            self.latent_world_model = ImagineWorldLatentDiT(config)
            self.reward_model = ImagineRewardModelSimple(config) # ImagineRewardModelSeg
        elif self.train_stage == 'stage5':
            self.ImagineWorld_model = ImagineWorldModel(config)
            self.latent_world_model = ImagineWorldLatentDiT(config)
            self.reward_model = ImagineRewardModelSeg(config) # ImagineRewardModelSeg
        #import pdb;pdb.set_trace()
        self.slice_indices = slice_indices
        self.config = config
        #import pdb;pdb.set_trace()
        if self.train_stage == 'stage2':
            self.initialize()
            self.freeze()
        
        if self.train_stage == 'stage3':
            self.initialize_stage_2()
            self.freeze_stage_2()
        
        if self.train_stage == 'stage4':
            self.initialize_stage_2()
            self.freeze_stage_2()
        
        if self.train_stage == 'stage5':
            self.initialize_stage_2()
            self.freeze_stage_2()

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__
    
    def freeze(self) -> None:
        # freeze stage1_model
        for param in self.ImagineWorld_model.parameters():
                param.requires_grad = False
    
    def freeze_stage_2(self) -> None:
        # freeze stage1_model
        #import pdb;pdb.set_trace()
        for param in self.ImagineWorld_model.parameters():
                param.requires_grad = False
        for param in self.latent_world_model.parameters():
                param.requires_grad = False        

    def initialize(self) -> None:
        """Inherited, see superclass."""
        #import pdb;pdb.set_trace()
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )["state_dict"]
        
        if "agent.ImagineWorld_model.trajectory_anchors" in state_dict:
            del state_dict["agent.ImagineWorld_model.trajectory_anchors"]
        #print("hahah")
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)
        print("success loaded stage1 model!")
    
    def initialize_stage_2(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(
                self._checkpoint_path, map_location=torch.device("cpu")
            )["state_dict"]
        #import pdb;pdb.set_trace()
        if "agent.ImagineWorld_model.trajectory_anchors" in state_dict:
            del state_dict["agent.ImagineWorld_model.trajectory_anchors"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)
        print("success loaded stage2 model!")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_tfu_sensors(self.slice_indices) 

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            ImagineWorldTargetBuilder(
                        trajectory_sampling=self._trajectory_sampling,
                        slice_indices=self.slice_indices,
                        sim_reward_dict_path=self.config.sim_reward_dict_path,
                        config=self.config,
                    ),
        ]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ImagineWorldFeatureBuilder(self.slice_indices, self.config)]

    def forward(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, torch.Tensor]:
        #import pdb;pdb.set_trace()
        result = self.ImagineWorld_model.forward_train(features, targets)
        #import pdb;pdb.set_trace()
        if self.train_stage == "stage2":
            result = self.latent_world_model(features, targets, result)
        elif self.train_stage == "stage3":
            result = self.latent_world_model.forward_test(features, targets, result)
            result = self.reward_model.forward_train(features, targets, result)
        elif self.train_stage == "stage4":
            result = self.latent_world_model.forward_test(features, targets, result)
            result = self.reward_model.forward_train(features, targets, result)
        elif self.train_stage == "stage5":
            result = self.latent_world_model.forward_test(features, targets, result)
            result = self.reward_model.forward_train(features, targets, result)
        return result
    
    def forward_test(self, features: Dict[str, torch.Tensor], token, targets=None) -> Dict[str, torch.Tensor]:
        #import pdb;pdb.set_trace()
        result = self.ImagineWorld_model.forward_test(features)
        result = self.latent_world_model.forward_test(features, targets, result)
        result = self.reward_model.forward_test(features, targets, result, token)
        return result
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return compute_ImagineWorld_loss(targets, predictions, self.config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        use_coslr_opt = self.config.use_coslr_opt if hasattr(self.config, 'use_coslr_opt') else False
        #import pdb;pdb.set_trace()
        if use_coslr_opt:
            return self.get_coslr_optimizers()
        else:
            return torch.optim.Adam(self.ImagineWorld_model.parameters(), lr=self._lr)
    
    def get_coslr_optimizers(self):
        optimizer_cfg = dict(type=self.config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self.config.weight_decay,
                            paramwise_cfg=self.config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self.config.scheduler_type,
                            milestones=self.config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        #import pdb;pdb.set_trace()
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self.ImagineWorld_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
            for k, v in self.latent_world_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v) # reward_model
            for k, v in self.reward_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self.ImagineWorld_model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})

        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=self.min_lr,
            epochs=self.max_epochs,
            warmup_epochs=3,
        )

        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, lr, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super(WarmupCosLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]