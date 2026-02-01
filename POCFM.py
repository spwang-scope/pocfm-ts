"""
Partially Observable Conditional Flow Matching (PO-CFM) for Time Series Forecasting

This implements the simplified algorithm (uniform timesteps) from:
"Conditional flow matching for partially observable time series prediction"

Key Components:
- ContextEncoder (f_ψ): Encodes observed sequence and external conditions
- VelocityNetwork (v_θ): Predicts velocity field conditioned on context
- POCFM: Full model combining both components

Data shapes:
- Stock data: [batch, seq_len, n_features] for observed, [batch, pred_len, n_features] for target
- Condition data: [batch, seq_len, n_cond_features]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for the flow time t."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size] or [batch_size, 1] tensor of time values in [0, 1]
        Returns:
            [batch_size, dim] embedding
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t * freqs.unsqueeze(0)  # [B, half_dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
            
        return embedding


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ContextEncoder(nn.Module):
    """
    Context Encoder (f_ψ): Encodes observed sequence and external conditions.
    
    Uses a Transformer encoder with cross-attention between observed data and conditions.
    
    Architecture:
    1. Project observed data and conditions to d_model dimension
    2. Add positional encoding
    3. Apply self-attention Transformer encoder
    4. Optional: Aggregate to fixed-size context or keep sequence
    """
    
    def __init__(
        self,
        n_features: int,
        n_cond_features: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        aggregate: str = 'none'  # 'none', 'mean', 'last'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.aggregate = aggregate
        
        # Input projections
        self.obs_proj = nn.Linear(n_features, d_model)
        self.cond_proj = nn.Linear(n_cond_features, d_model)
        
        # Combine observed and condition features
        self.combine = nn.Linear(2 * d_model, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x_obs: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_obs: [batch, seq_len, n_features] observed sequence
            c: [batch, seq_len, n_cond_features] condition sequence
            mask: [batch, seq_len] optional mask (1=valid, 0=masked)
            
        Returns:
            h_cond: [batch, seq_len, d_model] if aggregate='none'
                    [batch, d_model] if aggregate='mean' or 'last'
        """
        # Project inputs
        h_obs = self.obs_proj(x_obs)  # [B, L, d_model]
        h_cond = self.cond_proj(c)     # [B, L, d_model]
        
        # Combine observed and condition
        h = torch.cat([h_obs, h_cond], dim=-1)  # [B, L, 2*d_model]
        h = self.combine(h)  # [B, L, d_model]
        
        # Add positional encoding
        h = self.pos_encoder(h)
        
        # Transformer encoding
        h = self.transformer(h)
        h = self.norm(h)
        
        # Aggregate if specified
        if self.aggregate == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
                h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                h = h.mean(dim=1)
        elif self.aggregate == 'last':
            h = h[:, -1, :]
            
        return h


class VelocityNetwork(nn.Module):
    """
    Velocity Network (v_θ): Predicts velocity field conditioned on context.
    
    Architecture:
    1. Project noised target and concatenate with time embedding
    2. Use cross-attention to attend to context
    3. Predict velocity for each position in target sequence
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        context_dim = context_dim or d_model
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Input projection (noised target)
        self.input_proj = nn.Linear(n_features, d_model)
        
        # Positional encoding for target sequence
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Cross-attention layers for conditioning
        self.cross_attn_layers = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.norms3 = nn.ModuleList()
        
        for _ in range(n_layers):
            self.self_attn_layers.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            )
            self.cross_attn_layers.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                )
            )
            self.norms1.append(nn.LayerNorm(d_model))
            self.norms2.append(nn.LayerNorm(d_model))
            self.norms3.append(nn.LayerNorm(d_model))
        
        # Context projection (if dimensions differ)
        if context_dim != d_model:
            self.context_proj = nn.Linear(context_dim, d_model)
        else:
            self.context_proj = nn.Identity()
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features)
        )
        
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        h_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t: [batch, pred_len, n_features] noised target at time t
            t: [batch] or [batch, 1] flow time in [0, 1]
            h_cond: [batch, seq_len, context_dim] or [batch, context_dim] context encoding
            
        Returns:
            v: [batch, pred_len, n_features] predicted velocity
        """
        
        batch_size, pred_len, n_features = x_t.shape
        
        # Time embedding
        t_emb = self.time_embed(t)  # [B, d_model]
        t_emb = self.time_mlp(t_emb)  # [B, d_model]
        t_emb = t_emb.unsqueeze(1).expand(-1, pred_len, -1)  # [B, L, d_model]
        
        # Input projection
        h = self.input_proj(x_t)  # [B, pred_len, d_model]
        h = self.pos_encoder(h)
        
        # Add time embedding
        h = h + t_emb
        
        # Project context
        h_cond = self.context_proj(h_cond)
        if h_cond.dim() == 2:
            h_cond = h_cond.unsqueeze(1)  # [B, 1, d_model]
        
        # Transformer layers with cross-attention
        for i in range(len(self.self_attn_layers)):
            # Self-attention
            h_norm = self.norms1[i](h)
            h_attn, _ = self.self_attn_layers[i](h_norm, h_norm, h_norm)
            h = h + h_attn
            
            # Cross-attention to context
            h_norm = self.norms2[i](h)
            h_cross, _ = self.cross_attn_layers[i](h_norm, h_cond, h_cond)
            h = h + h_cross
            
            # Feed-forward
            h_norm = self.norms3[i](h)
            h = h + self.ffn_layers[i](h_norm)
        
        # Output projection
        v = self.output_proj(h)  # [B, pred_len, n_features]
        
        return v


class Model(nn.Module):
    """
    Partially Observable Conditional Flow Matching (PO-CFM) Model.
    
    Combines ContextEncoder and VelocityNetwork for time series forecasting
    using flow matching.
    
    Training:
        1. Encode context from observed sequence and conditions
        2. Sample flow time and noise
        3. Create OT interpolation path
        4. Predict velocity and compute flow matching loss
        
    Inference:
        1. Encode context
        2. Sample initial noise
        3. Integrate velocity field using Euler method
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Extract config
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_features = configs.enc_in  # Number of input features
        self.n_cond_features = getattr(configs, 'n_cond_features', configs.enc_in)
        
        d_model = getattr(configs, 'd_model', 128)
        n_heads = getattr(configs, 'n_heads', 4)
        e_layers = getattr(configs, 'e_layers', 2)
        d_layers = getattr(configs, 'd_layers', 3)
        d_ff = getattr(configs, 'd_ff', 256)
        dropout = getattr(configs, 'dropout', 0.1)
        
        # Inference parameters
        self.n_steps = getattr(configs, 'n_steps', 50)  # ODE integration steps
        self.sigma_min = getattr(configs, 'sigma_min', 1e-4)  # Minimum noise
        
        # Context Encoder
        self.encoder = ContextEncoder(
            n_features=self.n_features,
            n_cond_features=self.n_cond_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            aggregate='none'  # Keep sequence for cross-attention
        )
        
        # Velocity Network
        self.velocity = VelocityNetwork(
            n_features=self.n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=d_layers,
            d_ff=d_ff,
            dropout=dropout,
            context_dim=d_model
        )
        
    def compute_loss(
        self,
        x_obs: torch.Tensor,
        x_tar: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute flow matching loss.
        
        Args:
            x_obs: [batch, seq_len, n_features] observed sequence
            x_tar: [batch, pred_len, n_features] target sequence
            c: [batch, seq_len, n_cond_features] condition sequence
            mask: [batch, seq_len] optional mask for observed sequence
            
        Returns:
            loss: scalar loss value
            metrics: dict with additional metrics
        """
        batch_size = x_obs.shape[0]
        device = x_obs.device

        # Encode context
        # Remember to chech if nans are present
        assert not torch.isnan(x_obs).any(), "x_obs is NaN!"
        assert not torch.isnan(c).any(), "c is NaN!"
        h_cond = self.encoder(x_obs, c)  # [B, seq_len, d_model]
        h_cond = torch.nan_to_num(h_cond, nan=0.0, posinf=1, neginf=-1)
        assert not torch.isnan(h_cond).any(), "h_cond is NaN!"

        # Sample flow time t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        
        # Sample noise z_0 ~ N(0, I)
        #z_0 = torch.randn_like(x_tar)
        last_value = x_obs[:, -1:, :].expand(-1, self.pred_len, -1)  # [B, pred_len, n_features]
        #noise_scale = 0.5
        #z_0 = (1-noise_scale) * last_value + noise_scale * torch.randn_like(last_value)
        z_0 = last_value
        
        # OT interpolation: x_t = (1-t) * z_0 + t * x_tar
        t_expanded = t.view(batch_size, 1, 1)  # [B, 1, 1]
        x_t = (1 - t_expanded) * z_0 + t_expanded * x_tar

        # Target velocity: u_t = x_tar - z_0
        u_t = x_tar - z_0
        
        # Predict velocity
        assert not torch.isnan(x_t).any(), "x_t is NaN!"
        v_pred = self.velocity(x_t, t, h_cond)
        v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=1, neginf=-1)
        assert not torch.isnan(v_pred).any(), "Predicted velocity is NaN!"

        # Penalty loss:Enforce boundary value of close
        # violation=max(0,0.9*torch.abs(x_t[:, -1, 2] - x_tar[:, 0, 2]).mean().item())+max(0,1.1*torch.abs(x_t[:, -1, 2] - x_tar[:, -1, 2]).mean().item())
        
        # Flow matching loss
        loss = F.mse_loss(v_pred, u_t) # + 2*violation #+ 0.1*F.mse_loss(x_t[:, -1, :], x_tar[:, -1, :]) 
        
        metrics = {
            'loss': loss.item(),
            'v_norm': v_pred.norm(dim=-1).mean().item(),
            'u_norm': u_t.norm(dim=-1).mean().item()
        }

        assert not torch.isnan(loss), "Loss is NaN!"
        
        return loss, metrics
    
    @torch.no_grad()
    def sample(
        self,
        x_obs: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate predictions by integrating the velocity field.
        
        Args:
            x_obs: [batch, seq_len, n_features] observed sequence
            c: [batch, seq_len, n_cond_features] condition sequence
            mask: [batch, seq_len] optional mask
            n_steps: number of integration steps (default: self.n_steps)
            
        Returns:
            x_pred: [batch, pred_len, n_features] predicted sequence
        """
        n_steps = n_steps or self.n_steps
        batch_size = x_obs.shape[0]
        device = x_obs.device
        
        # Encode context
        h_cond = self.encoder(x_obs, c, mask)
        
        # Initialize from noise
        x_t = torch.randn(batch_size, self.pred_len, self.n_features, device=device)

        # x_obs statstics for denormalization
        #x_obs_stdev = torch.sqrt(torch.var(x_obs, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #x_obs_end = x_obs[:, -1:, :]
        
        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps # scale timestep to [0,1]
        for i in range(n_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            v = self.velocity(x_t, t, h_cond)
            x_t = x_t + dt * v
             

        return x_t
    
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass compatible with Time-Series-Library interface.
        
        During training, this returns the predicted sequence (for compatibility).
        The actual training uses compute_loss() method.
        
        Args:
            x_enc: [batch, seq_len, n_features] encoder input (observed)
            x_mark_enc: [batch, seq_len, n_mark_features] time features for encoder
            x_dec: [batch, label_len + pred_len, n_features] decoder input
            x_mark_dec: [batch, label_len + pred_len, n_mark_features] time features for decoder
            mask: optional mask
            
        Returns:
            predictions: [batch, pred_len, n_features]
        """
        # Use time marks as conditions
        c = x_mark_enc
        
        # If n_cond_features doesn't match, project
        if c.shape[-1] != self.n_cond_features:
            # Simple projection or padding
            if not hasattr(self, 'cond_adapter'):
                self.cond_adapter = nn.Linear(c.shape[-1], self.n_cond_features).to(c.device)
            c = self.cond_adapter(c)
        
        # Generate predictions
        return self.sample(x_enc, c, mask)


# Standalone functions for training and sampling (useful for custom training loops)

def flow_matching_loss(
    model: Model,
    x_obs: torch.Tensor,
    x_tar: torch.Tensor,
    c: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convenience function for computing flow matching loss."""
    loss, _ = model.compute_loss(x_obs, x_tar, c, mask)
    return loss


def generate_predictions(
    model: Model,
    x_obs: torch.Tensor,
    c: torch.Tensor,
    n_samples: int = 1,
    n_steps: int = 50
) -> torch.Tensor:
    """
    Generate multiple prediction samples.
    
    Args:
        model: POCFM model
        x_obs: [batch, seq_len, n_features] observed sequence
        c: [batch, seq_len, n_cond_features] conditions
        n_samples: number of samples to generate
        n_steps: integration steps
        
    Returns:
        samples: [n_samples, batch, pred_len, n_features]
    """
    samples = []
    for _ in range(n_samples):
        sample = model.sample(x_obs, c, n_steps=n_steps)
        samples.append(sample)
    return torch.stack(samples, dim=0)
