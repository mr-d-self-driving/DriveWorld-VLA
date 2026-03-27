from typing import Any, List, Dict, Union
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from torchvision import transforms
import math
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.agents.ImagineWorld.models.DiT import DiT_models
from navsim.agents.ImagineWorld.sampler.random_util import get_generator
from torchdiffeq import odeint_adjoint as odeint

def save_bev_activation_map(
    bev_feat,
    save_path,
    mode="l1",
    normalize=True,
    cmap="viridis",
    dpi=300
):
    """
    Save BEV activation / energy map as an image.

    Args:
        bev_feat: torch.Tensor [1, H, W, C]
        save_path: str, e.g. "bev_activation.png"
        mode: 'l1' | 'l2' | 'max'
        normalize: bool
        cmap: matplotlib colormap
        dpi: image resolution
    """

    assert bev_feat.ndim == 4 and bev_feat.shape[0] == 1
    bev_feat = bev_feat[0]  # [H, W, C]

    if mode == "l1":
        activation = bev_feat.abs().mean(dim=-1)
    elif mode == "l2":
        activation = torch.sqrt((bev_feat ** 2).sum(dim=-1))
    elif mode == "max":
        activation = bev_feat.abs().max(dim=-1)[0]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    activation = activation.detach().cpu()

    if normalize:
        activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(activation, cmap=cmap)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

    return activation
# 使用示例（你可以直接跑）
# save_bev_activation_map(
#     bev_feat,
#     save_path="figs/bev_activation_l1.png",
#     mode="l1"
# )

scale_factor = 0.18215
reward_weights = [0.1, 0.5, 0.5, 1.0]
img_w = 8
img_h = 8
img_c = 256
class ImagineWorldLatentDiT(nn.Module):
    def __init__(self, 
                 config,
                ):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(256)

        self.layernorm_2 = nn.LayerNorm(256)
        self.config = config

        self.dit = DiT_models[config.dit_type](
            img_resolution=config.dit_img_resolution,
            in_channels=config.dit_in_channels,
            label_dropout=config.dit_label_dropout,
            num_classes=config.dit_num_classes,
        )

    def weighted_reward_calculation(self, im_rewards, sim_rewards) -> torch.Tensor:
        """
        Calculate the final reward for each trajectory based on the given weights.

        Args:
            im_rewards (torch.Tensor): Imitation rewards for each trajectory. Shape: [batch_size, num_traj]
            sim_rewards (List[torch.Tensor]): List of metric rewards for each trajectory. Each tensor shape: [batch_size, num_traj]
            w (List[float]): List of weights for combining the rewards.

        Returns:
            torch.Tensor: Final weighted reward for each trajectory. Shape: [batch_size, num_traj]
        """
        #import pdb;pdb.set_trace()
        w = reward_weights
        S_NC = sim_rewards[:,0,:]
        S_DAC = sim_rewards[:,1,:]
        S_EP = sim_rewards[:,2,:]
        S_TTC = sim_rewards[:,3,:]
        S_COMFORT = sim_rewards[:,4,:]
        #import pdb;pdb.set_trace()
        #S_NC, S_DAC, S_EP, S_TTC, S_COMFORT = S_NC.squeeze(-1), S_DAC.squeeze(-1), S_EP.squeeze(-1), S_TTC.squeeze(-1), S_COMFORT.squeeze(-1)
        #self.metric_keys = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound', 'comfort']
        # Calculate assembled cost based on the provided formula
        assembled_cost = (
            w[0] * torch.log(im_rewards) +
            w[1] * torch.log(S_NC) +
            w[2] * torch.log(S_DAC) +
            w[3] * torch.log(5 * S_TTC + 2 * S_COMFORT + 5 * S_EP)
        )
        #import pdb;pdb.set_trace()
        return assembled_cost
    
    def forward(self, features, targets, results):
        pred_results = {}
        future_flatten_bev_feature = results['future_flatten_bev_feature_1']
        future_flatten_bev_feature = self.layernorm_1(future_flatten_bev_feature) * scale_factor#重点画
        B = future_flatten_bev_feature.shape[0]
        C = 256
        H = 8
        W = 8
        future_bev_feature = future_flatten_bev_feature.permute(0,2,1).view(B, C, H, W)
        flatten_bev_feature = results['flatten_bev_feature_1']
        flatten_bev_feature = self.layernorm_2(flatten_bev_feature) * scale_factor
        bev_feature = flatten_bev_feature.permute(0,2,1).view(B, C, H, W)

        t = torch.rand((future_flatten_bev_feature.shape[0],), dtype=future_flatten_bev_feature.dtype, device=future_flatten_bev_feature.device)
        y = targets['trajectory'].squeeze(1)[...,:2].to(torch.bfloat16) # [1, 1, 8, 3]
        
        z1 = future_bev_feature      # data
        z0 = torch.randn_like(z1)    # noise
        t = torch.rand((B,), device=z1.device).view(B,1,1,1)
        eps = 1e-5
        # interpolation
        z_t = (1 - t) * z0 + (eps + (1 - eps) * t) * z1
        # ground truth vector field
        u = (1 - eps) * z1 - z0

        v_pred = self.dit(t.squeeze(), z_t, y, bev_feature)
        pred_results["v_gt"] = u
        pred_results["v_pred"] = v_pred
        #import pdb;pdb.set_trace()
        
        return pred_results #features
    
    def sample_from_model(self, model, x_0, model_kwargs, args):
        #options = {"dtype": torch.float32,}
        options = {"step_size": args.step_size/100, "perturb": args.perturb}
        t = torch.tensor([0.0, 1.0], device="cuda")

        def denoiser(t, x_0):
            return model.forward_with_cfg(t, x_0, **model_kwargs)
        #import pdb;pdb.set_trace()
        fake_image = odeint(
            denoiser,
            x_0,
            t,
            method=args.method,
            atol=args.atol,
            rtol=args.rtol,
            adjoint_method=args.method,
            adjoint_atol=args.atol,
            adjoint_rtol=args.rtol,
            options=options,
            adjoint_params=model.parameters(),
        )
        return fake_image
    
    def fast_sample_from_model(self, model, x_0, model_kwargs, args):
        pass
    
    def forward_test(self, features, targets, results):
        #results = {}
        trajectory_anchors = results['trajectory_anchors']
        trajectory_offset = results['trajectory_offset']
        n_samples = 16 # 在一个batch_size中我们选择得分为前16的轨迹进行未来想象
        bs = trajectory_offset.shape[0]
        expand_trajectory_anchors = trajectory_anchors.unsqueeze(0).repeat(bs, 1, 1, 1)
        output_trajectory = expand_trajectory_anchors + trajectory_offset
        im_rewards = results['im_rewards']
        sim_rewards = results['sim_rewards']
        final_rewards = self.weighted_reward_calculation(im_rewards, sim_rewards)
        results.update({'final_reward': final_rewards})
        topk_indices = torch.topk(final_rewards, k=16, dim=1).indices
        #import pdb;pdb.set_trace()
        topk_trajectories = torch.gather(output_trajectory, dim=1, index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 3))
        flatten_bev_feature = results['flatten_bev_feature_1']
        with torch.no_grad():
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
            generator = get_generator("determ", bs*n_samples, 42+rank)
            x0 = generator.randn(bs*n_samples, img_c, img_h, img_w) # b,8,2
            cfg_x0 = torch.cat([x0, x0], 0).to(flatten_bev_feature.device)
            xy_topk_trajectories = topk_trajectories[...,:2].view(bs*n_samples, 8, 2)
            xy0_topk_trajectories = torch.zeros_like(xy_topk_trajectories)
            condition_y0 = torch.cat([xy_topk_trajectories, xy0_topk_trajectories], 0)
            flatten_bev_feature = self.layernorm_2(flatten_bev_feature) * scale_factor
            flatten_bev_feature = flatten_bev_feature.unsqueeze(1).repeat(1, n_samples, 1, 1)
            bev_feature = flatten_bev_feature.permute(0, 1, 3, 2).view(bs*n_samples, img_c, img_h, img_w)
            condition_y1 = torch.cat([bev_feature, bev_feature], 0)
            model_kwargs = dict(y=condition_y0, y1=condition_y1, cfg_scale=self.config.cfg_scale)
            future_sample = self.sample_from_model(self.dit, cfg_x0, model_kwargs, self.config)[-1]
            results.update({'latent_future': future_sample})
        #import pdb;pdb.set_trace()
        return results
