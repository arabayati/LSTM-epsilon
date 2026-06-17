#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

class EpsilonStateResetModel(nn.Module):
    def __init__(self, input_dim, hidden_size=256, n_mul=16, dropout_rate=0.4):
        super(EpsilonStateResetModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_mul = n_mul
        
        # 1. The Core Context Encoder
        # Using standard CuDNN LSTM for maximum speed across the 365-day sequence
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 2. Dynamic Head (Predicts epsilon_t for all components)
        self.eps_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, n_mul)
        )
        
        # 3. Peak Head (Predicts the base flow Q0 to solve the initialization dilemma)
        self.peak_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, n_mul)
        )
        
        # 4. Static Head (Predicts Alpha, LP, Gamma based on the final hidden state)
        # Outputs 3 values per component: [alpha, lp, gamma] -> total 3 * n_mul
        self.static_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3 * n_mul),
            nn.Sigmoid() # Squashes to [0,1] before bounds scaling
        )

    ## def forward
    def forward(self, z_seq, pet_seq, sm_seq, rec_mask, start_mask, bounds, bufftime=365):
        """
        z_seq: [Time_Total, Batch, InputDim] - Features (Dynamic + Static)
        pet_seq: [Time_Total, Batch, 1] - Physical PET
        sm_seq: [Time_Total, Batch, 1] - Physical Soil Moisture
        rec_mask: [Time_Eval, Batch, 1] - 1 if day is an eligible recession
        start_mask: [Time_Eval, Batch, 1] - 1 if day is the FIRST day of a recession
        bounds: [Batch, 6] - [alpha_min, alpha_max, lp_min, lp_max, gm_min, gm_max]
        """
        # --- A. Context Processing ---
        lstm_out, _ = self.lstm(z_seq)  # [Time_Total, Batch, Hidden] (e.g., 730 days)
        lstm_out = self.dropout(lstm_out)
        
        # ?? EXACT FIX: Slice away the warmup period here ??
        lstm_out = lstm_out[bufftime:, :, :]  # Now strictly [Time_Eval, Batch, Hidden] (e.g., 365)
        pet_seq  = pet_seq[bufftime:, :, :]   # Now strictly [Time_Eval, Batch, 1]
        sm_seq   = sm_seq[bufftime:, :, :]    # Now strictly [Time_Eval, Batch, 1]
        
        # Grab the dimensions AFTER slicing so the ODE loop runs for the correct number of days
        Time, Batch, _ = lstm_out.shape       # Time is now 365
        
        # --- B. Parameter Generation ---
        # 1. Dynamic Epsilon (must be strictly positive, scaled down initially for physical realism)
        raw_eps = self.eps_head(lstm_out)
        eps_t = F.softplus(raw_eps)  #* 0.01  # [Time, Batch, N_mul]
        
        # 2. Dynamic Q_base (must be strictly positive)
        raw_qbase = self.peak_head(lstm_out)
        q_base_t = F.softplus(raw_qbase)    # [Time, Batch, N_mul]
        

        # 3. Static Parameters from the last hidden state of the evaluated sequence
        # This follows the static LSTM-HBV-style parameterization.
        h_final = lstm_out[-1, :, :]       # [Batch, Hidden]
        static_raw = self.static_head(h_final).view(Batch, self.n_mul, 3) # [Batch, N_mul, 3]
        
        # Unpack bounds: shape is [Batch, 1] to broadcast across the N_mul components
        a_min, a_max = bounds[:, 0:1], bounds[:, 1:2]
        l_min, l_max = bounds[:, 2:3], bounds[:, 3:4]
        g_min, g_max = bounds[:, 4:5], bounds[:, 5:6]
        
        alpha = a_min + (a_max - a_min) * static_raw[:, :, 0] # [Batch, N_mul]
        lp    = l_min + (l_max - l_min) * static_raw[:, :, 1] # [Batch, N_mul]
        gamma = g_min + (g_max - g_min) * static_raw[:, :, 2] # [Batch, N_mul]
        
        # --- C. Vectorized AET Calculation ---
        # Evaluate AET for the target 365 days and all 16 components instantly
        sm_term = torch.clamp(sm_seq / (lp.unsqueeze(0) + EPS), min=EPS) # [Time, Batch, N_mul]
        aet_t = pet_seq * torch.pow(sm_term, gamma.unsqueeze(0))         # [Time, Batch, N_mul]
        aet_t = torch.clamp(aet_t, max=pet_seq) # AET cannot exceed PET
        
        # --- D. The State-Reset Differentiable Physics Engine ---
        q_out_list = []
        q_prev = q_base_t[0] # Initialize first step
        
        #alpha_eff = alpha.unsqueeze(0) # [1, Batch, N_mul]
        alpha_eff = alpha # [Batch, N_mul]  
        
        for t in range(Time):
            # 1. State Reset: If today is a 'start' day, discard q_prev and pull Q_base from yesterday
            reset_val = q_base_t[t-1] if t > 0 else q_base_t[0]
            is_start = start_mask[t] # [Batch, 1]
            q_curr = torch.where(is_start > 0.5, reset_val, q_prev)
            
            # 2. EXACT Piecewise Closed-Form Evaluation
            # Let b_t = epsilon, and a_t = epsilon * alpha * AET
            b_t = eps_t[t]                           # [Batch, N_mul]
            a_t = b_t * (alpha_eff * aet_t[t])       # [Batch, N_mul]
            
            # Formula A: Exact Integral WITH AET
            # Q_{t+1} = (a * Q) / [ (b * Q + a) * exp(a) - b * Q ]
            denom = (b_t * q_curr + a_t) * torch.exp(a_t) - (b_t * q_curr)
            q_next_aet = (a_t * q_curr) / torch.clamp(denom, min=EPS)
            
            # Formula B: Exact Integral WITHOUT AET (Limit as a -> 0)
            # Q_{t+1} = Q / (1 + b * Q)
            q_next_zero_aet = q_curr / (1.0 + b_t * q_curr)
            
            # Safely switch based on whether the AET term (a_t) is practically zero
            q_next = torch.where(a_t < 1e-6, q_next_zero_aet, q_next_aet)
            
            # 3. Mask Forward: If we are not in a recession, keep the state tracked to Q_base
            is_rec = rec_mask[t] # [Batch, 1]
            q_prev = torch.where(is_rec > 0.5, q_next, q_base_t[t])
            
            q_out_list.append(q_prev)
            
        # Stack back into [Time, Batch, N_mul]
        q_hat_components = torch.stack(q_out_list, dim=0)
        
        # --- E. Ensemble Collapse ---
        # Average the 16 components to get the final predicted trajectory
        q_hat_total = torch.mean(q_hat_components, dim=-1, keepdim=True) # [Time, Batch, 1]
        
        return {
            'q_hat': q_hat_total,            # The final prediction [T, B, 1]
            'q_components': q_hat_components,# The 16 individual paths [T, B, 16]
            'q_base': q_base_t,              # The predicted continuous Q0 [T, B, 16]
            'eps': eps_t,                    # The dynamic epsilon [T, B, 16]
            'aet': aet_t,                    # The calculated AET [T, B, 16]
            'alpha': alpha,                  # The static alpha [B, 16]
            'lp': lp,                        # The static lp [B, 16]
            'gamma': gamma                   # The static gamma [B, 16]
        }