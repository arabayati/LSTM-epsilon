#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

EPS = 1e-6

def huber_loss(pred, target, delta=0.5):
    """Numerically stable Huber loss."""
    err = torch.abs(pred - target)
    quad = torch.clamp(err, max=delta)
    lin = err - quad
    return 0.5 * quad**2 + delta * lin

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_path=1.2, lambda_rhs=1.0, lambda_smooth=20.0, lambda_q0=2.0, delta=0.5):
        super(PhysicsInformedLoss, self).__init__()
        self.l_path = lambda_path
        self.l_rhs = lambda_rhs
        self.l_smooth = lambda_smooth
        self.l_q0 = lambda_q0
        self.delta = delta

    def forward(self, model_out, obs_q, rec_mask, start_mask):
        """
        model_out: Dictionary returned by EpsilonStateResetModel.forward()
        obs_q: [Time, Batch, 1] - Observed Streamflow
        rec_mask: [Time, Batch, 1] - Boolean mask for valid recession days
        start_mask: [Time, Batch, 1] - Boolean mask for recession start days
        """
        
        # Unpack model outputs
        q_hat = model_out['q_hat']               # [T, B, 1]
        q_base = model_out['q_base']             # [T, B, 16]
        eps = model_out['eps']                   # [T, B, 16]
        aet = model_out['aet']                   # [T, B, 16]
        alpha = model_out['alpha'].unsqueeze(0)  # [1, B, 16]
        
        # Shape safety checks
        if q_hat.shape != obs_q.shape:
            raise RuntimeError(f"q_hat shape {q_hat.shape} does not match obs_q shape {obs_q.shape}")

        if rec_mask.shape != obs_q.shape:
            raise RuntimeError(f"rec_mask shape {rec_mask.shape} does not match obs_q shape {obs_q.shape}")

        if start_mask.shape != obs_q.shape:
            raise RuntimeError(f"start_mask shape {start_mask.shape} does not match obs_q shape {obs_q.shape}")

        if eps.ndim != 3 or aet.ndim != 3 or q_base.ndim != 3:
            raise RuntimeError(
                f"Expected eps/aet/q_base to be 3D [T,B,n_mul], got "
                f"eps={eps.shape}, aet={aet.shape}, q_base={q_base.shape}"
            )
        
        # Ensure obs_q doesn't contain NaNs where we intend to calculate loss
        obs_q_safe = torch.nan_to_num(obs_q, nan=0.0)
        obs_q_log = torch.log(torch.clamp(obs_q_safe, min=EPS))
        q_hat_log = torch.log(torch.clamp(q_hat, min=EPS))

        # -------------------------------------------------------------
        # 1. Primary Loss (L_path): Match the integrated trajectory
        # -------------------------------------------------------------
        path_err = huber_loss(q_hat_log, obs_q_log, self.delta)
        # Apply mask: only care about path error on valid recession days
        valid_rec_days = torch.sum(rec_mask).clamp_min(1.0)
        loss_path = torch.sum(path_err * rec_mask) / valid_rec_days

        # -------------------------------------------------------------
        # 2. Secondary Loss (L_RHS): Local derivative consistency 
        # (Anchors the physics strictly to observed Q)
        # -------------------------------------------------------------
        # Calculate observed dQ/dt
        obs_q_t = obs_q_safe[:-1, :, :] # [T-1, B, 1]
        obs_q_tp1 = obs_q_safe[1:, :, :]# [T-1, B, 1]
        dQ_obs = obs_q_tp1 - obs_q_t
        
        # Calculate predicted instantaneous dQ/dt based strictly on obs_q
        # Broadcast obs_q to the 16 components
        # Calculate predicted instantaneous dQ/dt based strictly on obs_q
        # Broadcast obs_q to however many ODE components the model is using
        aet_t = aet[:-1, :, :]
        eps_t = eps[:-1, :, :]
        
        n_mul = eps_t.shape[-1]
        obs_q_t_comp = obs_q_t.repeat(1, 1, n_mul)
        
        # Evaluated RHS: dQ = -eps * Q_{obs}^2 - alpha * AET * Q_{obs}
        dQ_pred_components = -eps_t * (obs_q_t_comp**2 + (alpha * aet_t) * obs_q_t_comp)
        
        # Collapse the 16 predicted derivatives to match the 1 observed derivative
        dQ_pred = torch.mean(dQ_pred_components, dim=-1, keepdim=True) # [T-1, B, 1]
        
        rhs_err = huber_loss(dQ_pred, dQ_obs, self.delta)
        # Align mask (shift by 1 because dQ evaluates step t to t+1)
        rec_mask_shifted = (rec_mask[:-1, :, :] > 0.5) & (rec_mask[1:, :, :] > 0.5)
        valid_rhs_days = torch.sum(rec_mask_shifted).clamp_min(1.0)
        loss_rhs = torch.sum(rhs_err * rec_mask_shifted) / valid_rhs_days

        # -------------------------------------------------------------
        # 3. Smoothness Loss (L_smooth): Prevent erratic jumping in Epsilon
        # -------------------------------------------------------------
        eps_mean = torch.mean(eps, dim=-1, keepdim=True) # [T, B, 1]
        if eps_mean.shape[0] >= 3:
            eps_0 = eps_mean[:-2, :, :]
            eps_1 = eps_mean[1:-1, :, :]
            eps_2 = eps_mean[2:, :, :]
            second_deriv = eps_2 - 2 * eps_1 + eps_0
            # Only penalize jaggedness if the whole 3-day window is a recession
            smooth_mask = (rec_mask[:-2] > 0.5) & (rec_mask[1:-1] > 0.5) & (rec_mask[2:] > 0.5)
            valid_smooth = torch.sum(smooth_mask).clamp_min(1.0)
            loss_smooth = torch.sum((second_deriv**2) * smooth_mask) / valid_smooth
        else:
            loss_smooth = torch.tensor(0.0, device=eps.device)

        # -------------------------------------------------------------
        # 4. Q0 Anchor Loss (L_Q0): Ensure the Peak Head is accurate
        # -------------------------------------------------------------
        q_base_mean = torch.mean(q_base, dim=-1, keepdim=True) # [T, B, 1]
        q_base_log = torch.log(torch.clamp(q_base_mean, min=EPS))
        
        q0_err = huber_loss(q_base_log, obs_q_log, self.delta)
        
        # ?? FIX: Shift the start_mask backward by 1 day. 
        # If recession starts on Day t, we must supervise the Q0 prediction from Day t-1.
        start_mask_q0 = torch.zeros_like(start_mask)
        start_mask_q0[:-1, :, :] = start_mask[1:, :, :]
        
        valid_starts = torch.sum(start_mask_q0).clamp_min(1.0)
        loss_q0 = torch.sum(q0_err * start_mask_q0) / valid_starts
        # -------------------------------------------------------------
        # Total Loss Assembly
        # -------------------------------------------------------------
        total_loss = (self.l_path * loss_path) + \
                     (self.l_rhs * loss_rhs) + \
                     (self.l_smooth * loss_smooth) + \
                     (self.l_q0 * loss_q0)

        # Return a dictionary for easy logging in train_main.py
        return {
            'total': total_loss,
            'l_path': loss_path,
            'l_rhs': loss_rhs,
            'l_smooth': loss_smooth,
            'l_q0': loss_q0
        }