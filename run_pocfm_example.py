"""
Minimal Working Example: Partially Observable Conditional Flow Matching (PO-CFM)
for Stock Time Series Prediction

This script demonstrates training and inference with dummy data.

Data Shapes:
- Stock data: [T_total, N_stocks, N_indicator]
- Condition data: [T_total, N_indicator]

The model learns to predict future stock indicators given:
1. Historical observations of a single stock
2. Market-wide condition data (e.g., indices, sentiment)
"""

#from networkx import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os
import sys
import pandas as pd
import random
import pickle
from sklearn.preprocessing import StandardScaler
from torchinfo import summary

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from POCFM import Model, ContextEncoder, VelocityNetwork

np.random.seed(9487)
torch.manual_seed(9487)

# =============================================================================
# Configuration
# =============================================================================

class Config:
    """
    Configuration for PO-CFM model.
    
    #using_train_exp: Added additional attributes required by Exp_POCFM.train()
    """
    
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        n_features: int = 6,        # N_indicator for stocks
        n_cond_features: int = 10,   # N_indicator for conditions
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        n_steps: int = 50,          # ODE integration steps
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        train_epochs: int = 10,
        use_gpu: bool = True,
        gpu: int = 0,
        save_path: str = 'pocfm_stock/',
        scaler_path: str = 'pocfm_stock/scaler.pkl',
        #using_train_exp: Added patience for early stopping (used in exp_pocfm.train())
        patience: int = 10,
        #using_train_exp: Added checkpoints path (used in exp_pocfm.train())
        checkpoints: str = './checkpoints/',
        #using_train_exp: Added use_multi_gpu flag (used in exp_pocfm._build_model())
        use_multi_gpu: bool = False,
        #using_train_exp: Added devices string (used in exp_pocfm._acquire_device())
        devices: str = '0',
        #using_train_exp: Added device_ids list (used in exp_pocfm._build_model())
        device_ids: list = None
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.enc_in = n_features
        self.n_cond_features = n_cond_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.gpu = gpu
        self.save_path = save_path
        self.scaler_path = scaler_path
        #using_train_exp: Initialize additional attributes for exp_pocfm compatibility
        self.patience = patience
        self.checkpoints = checkpoints
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices
        self.device_ids = device_ids if device_ids is not None else [0]


# =============================================================================
# Dummy Data Generation
# =============================================================================

def generate_dummy_stock_data(
    T_total: int = 1000,
    N_stocks: int = 50,
    N_indicator: int = 5,
    N_cond_indicator: int = 10,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic stock data with market conditions.
    
    The data simulates:
    - Stock indicators: price returns, volume, volatility, momentum, sentiment
    - Market conditions: index values, sector performance, economic indicators
    
    Returns:
        stock_data: [T_total, N_stocks, N_indicator]
        cond_data: [T_total, N_cond_indicator]
    """
    t = np.linspace(0, 10 * np.pi, T_total)
    
    # Generate market-wide factors (conditions)
    cond_data = np.zeros((T_total, N_cond_indicator))
    
    # Market trends (various frequencies)
    for i in range(N_cond_indicator):
        freq = 0.5 + i * 0.3
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        cond_data[:, i] = amplitude * np.sin(freq * t + phase)
        cond_data[:, i] += noise_level * np.random.randn(T_total)
    
    # Generate stock data influenced by market conditions
    stock_data = np.zeros((T_total, N_stocks, N_indicator))
    
    for stock_idx in range(N_stocks):
        # Each stock has its own sensitivity to market factors
        beta = np.random.randn(N_cond_indicator, N_indicator) * 0.3
        
        # Base influence from market conditions
        base_signal = cond_data @ beta
        
        # Stock-specific dynamics
        for ind_idx in range(N_indicator):
            # Add stock-specific patterns
            freq = 1.0 + stock_idx * 0.05
            phase = stock_idx * 0.2
            stock_specific = 0.5 * np.sin(freq * t + phase)
            
            # Combine market influence and stock-specific
            stock_data[:, stock_idx, ind_idx] = (
                base_signal[:, ind_idx] + 
                stock_specific +
                noise_level * np.random.randn(T_total)
            )
    
    return stock_data.astype(np.float32), cond_data.astype(np.float32)


class StockDataset(Dataset):
    """
    Dataset for stock prediction with market conditions.
    
    Each sample contains:
    - x_obs: [seq_len, n_features] observed sequence for one stock
    - x_tar: [pred_len, n_features] target sequence to predict
    - c: [seq_len, n_cond_features] market condition sequence
    """
    
    def __init__(
        self,
        stock_data: np.ndarray,
        cond_data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 24,
        stride: int = 10,
        mode: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        scaler_path: str = 'pocfm_stock/scaler.pkl'
    ):
        """
        Args:
            stock_data: [T_total, N_indicator, N_stocks]
            cond_data: [T_total, N_cond_indicator]
            seq_len: length of input sequence
            pred_len: length of prediction horizon
            stride: step size between consecutive samples
            mode: 'train', 'val', or 'test'
            train_ratio: fraction of data for training
            val_ratio: fraction of data for validation
        """
        self.scaler = None
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.total_len = seq_len + pred_len
        
        # Total time length, number of stocksnumber of indicators, 
        self.T_total, self.N_indicator, self.N_stocks = stock_data.shape
        
        # Split data temporally
        train_end = int(self.T_total * train_ratio)
        val_end = int(self.T_total * (train_ratio + val_ratio))
        
        if mode == 'train':
            t_start, t_end = 0, train_end
        elif mode == 'val':
            t_start, t_end = train_end, val_end
        elif mode == 'test':  # test
            t_start, t_end = val_end, self.T_total
        else:   # standalone prediction on all data
            t_start, t_end = 0, self.T_total  # use all data

        if mode == 'train':
            # only fit on training split
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            # Transpose: [T, N_indicator, N_stocks] -> [T, N_stocks, N_indicator]
            # Reshape:   [T, N_stocks, N_indicator] -> [T * N_stocks, N_indicator]
            X_train_2d = stock_data[t_start:t_end].transpose(0, 2, 1).reshape(-1, self.N_indicator)
            X_train_scaled_2d = self.scaler.fit_transform(X_train_2d)
            # Save scaler for future use
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to {scaler_path}")
        elif (mode == 'test') or (mode == 'val'):    # within dataset
            try:
                with open(scaler_path, 'rb') as file:
                    # Load the object from the file
                    self.scaler = pickle.load(file)
                    print("Scaler successfully loaded.")
            except Exception as e:
                print("Error loading scaler:", e)
            self.scaler.transform(stock_data[t_start:t_end].transpose(0, 2, 1).reshape(-1, self.N_indicator))
        elif mode == 'standalone':   # outside dataset (future data)
            # fit whole input for standalone prediction
            try:
                with open(scaler_path, 'rb') as file:
                    # Load the object from the file
                    self.scaler = pickle.load(file)
                    print("Scaler successfully loaded.")
            except Exception as e:
                print("Error loading scaler:", e)
            self.scaler.transform(stock_data[t_start:t_end].transpose(0, 2, 1).reshape(-1, self.N_indicator))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Transform ALL data with correct transpose
        X_all_2d = stock_data.transpose(0, 2, 1).reshape(-1, self.N_indicator)
        X_scaled_2d = self.scaler.transform(X_all_2d)

        # Reshape back: [T * N_stocks, N_indicator] -> [T, N_stocks, N_indicator]
        # Transpose back: [T, N_stocks, N_indicator] -> [T, N_indicator, N_stocks]
        stock_data = X_scaled_2d.reshape(self.T_total, self.N_stocks, self.N_indicator).transpose(0, 2, 1)

        assert(self.scaler is not None), "Scaler must be initialized."

        self.stock_data = stock_data[t_start:t_end,:,:]
        self.cond_data = cond_data[t_start:t_end,:]
        T = self.stock_data.shape[0]
        
        # Create indices for all valid windows across all stocks
        self.samples = []
        for t in range(0, T - self.total_len + 1, stride):
            for stock_idx in range(self.N_stocks):
                sample_seq = self.stock_data[t:t + self.seq_len, :, stock_idx]  # [seq_len, N_indicator]
                zero_seq = (not (sample_seq[0].all() == sample_seq[-1].all()) or sample_seq[0].sum() == 0 or sample_seq[-1].sum() == 0)
                #activities = np.abs(sample_seq[:,2].max() - sample_seq[:,2].min()) / ( sample_seq[:,2].min() + 1e-5) > 0.35
                max_diff = np.abs(max(sample_seq[:,2]) - min(sample_seq[:,2]))
                act = max_diff/min(sample_seq[:,2])
                volatilities = self.compute_volatility(sample_seq)
                if not zero_seq:
                    
                    if mode == 'standalone' or mode == 'test' or mode=='val':
                        self.samples.append((t, stock_idx))
                    elif mode == 'train':
                        #if (max_diff>10) or (act>0.1):   # threshold for volatility
                            self.samples.append((t, stock_idx))
        assert  (len(self.samples) > 0), 'No samples found for mode: {mode}'
        print(f"Sample number under {mode}: {len(self.samples)}")

                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        t, stock_idx = self.samples[idx]
        
        # Extract sequences
        x_obs = self.stock_data[t:t + self.seq_len, :, stock_idx]
        x_tar = self.stock_data[t + self.seq_len:t + self.total_len, :, stock_idx]
        c = self.cond_data[t:t + self.seq_len, :]
        
        return (
            torch.from_numpy(x_obs),
            torch.from_numpy(x_tar),
            torch.from_numpy(c)
        )
    
    def inverse_transform_stock_data(self, scaled_data):
        """
        Inverse transform that only used in standalone prediction mode.
        """
        original_shape = scaled_data.shape

        # temporarily broken warning

        # Backup original mean in self.scaler
        original_mean = self.scaler.mean_

        # select indicator axis
        if len(original_shape) == 3:
            # indicators for stocks data at axis=1
            x_obs_last = scaled_data[:, -1]  # last observed value for anchoring
            self.scaler.mean_ = x_obs_last
        elif len(original_shape) == 2:
            # standalone predictions: indicators at axis=1
            x_obs_last = scaled_data[:, -1]  # last observed value for anchoring
            self.scaler.mean_ = x_obs_last


        # Get the column of indicators indices in dataset
        
        #scaled_data[indi].mean(axis=0)
        
        if len(original_shape) == 2:
            # Simple 2D case: [T, N_indicator] or [seq_len, N_indicator]
            #full_scaled_data = torch.cat([x_obs, scaled_data], dim=0)
            original_data = self.scaler.inverse_transform(scaled_data)
        
        else:
            raise ValueError(f"Unexpected data shape: {original_shape}")
        
        # Restore original mean in self.scaler
        self.scaler.mean_ = original_mean
        
        return original_data


    def inverse_transform_stock_tensor(self, scaled_tensor):
        """
        Inverse transform scaled stock tensor back to original scale.
        
        Args:
            scaled_tensor: PyTorch tensor with shape:
                        - [T, N_indicator, N_stocks] for full stock data
                        - [batch, seq_len, N_indicator] for model predictions
        
        Returns:
            Tensor in original scale with same shape as input
        """
        was_tensor = isinstance(scaled_tensor, torch.Tensor)
        
        if was_tensor:
            device = scaled_tensor.device
            scaled_data = scaled_tensor.cpu().numpy()
        else:
            device = None
            scaled_data = scaled_tensor
        
        # Apply inverse transform
        original_data = self.inverse_transform_stock_data(scaled_data)
        
        # Convert back to tensor if needed
        #if was_tensor:
        #    return torch.from_numpy(original_data.astype(np.float32)).to(torch.device)
        return original_data

    def transform_new_data(new_data, scaler):
        """Transform new data using existing scaler"""
        return scaler.transform(new_data)

    def inverse_transform_predictions(predictions, scaler):
        """Inverse transform predictions"""
        return scaler.inverse_transform(predictions)
    
    def compute_volatility(self,x_obs):
        """
        Compute volatility within the observation window.
        
        Args:
            x_obs: array of shape (seq_len,) or (seq_len, features) - price series
            method: 'log_return' or 'simple_return'
        
        Returns:
            volatility: scalar or array of volatilities per feature
        """
        seq_len = x_obs.shape[0]

        # Handle multi-dimensional input
        if x_obs.ndim == 1:
            x_obs = x_obs.reshape(-1, 1)
        
        # Avoid division by zero
        x_obs = np.clip(x_obs, 1e-8, None)
        
        # Simple returns
        returns = np.diff(x_obs, axis=0) / x_obs[:-1]
        
        # Standard deviation of returns
        volatility = np.std(returns, axis=0)

        return volatility*np.sqrt(seq_len)


# =============================================================================
# Training Functions
# =============================================================================

#using_train_exp: Preserved train_epoch() but not called in main training flow
def train_epoch(
    model: Model,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    #using_train_exp: This function is preserved but NOT used in main training.
    The main training now uses Exp_POCFM.train() from exp_pocfm.py instead.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x_obs, x_tar, c) in enumerate(dataloader):
        x_obs = x_obs.to(device)
        x_tar = x_tar.to(device)
        c = c.to(device)
        
        optimizer.zero_grad()
        loss, metrics = model.compute_loss(x_obs, x_tar, c)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def evaluate(
    model: Model,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for x_obs, x_tar, c in dataloader:
            x_obs = x_obs.to(device)
            x_tar = x_tar.to(device)
            c = c.to(device)
            
            # Compute flow matching loss
            loss, _ = model.compute_loss(x_obs, x_tar, c)
            total_loss += loss.item()
            
            # Generate predictions
            preds = model.sample(x_obs, c)
            
            all_preds.append(preds.cpu().numpy())
            all_trues.append(x_tar.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    mse = np.mean((all_preds - all_trues) ** 2)
    mae = np.mean(np.abs(all_preds - all_trues))
    
    return total_loss / len(dataloader), mse, mae


#using_train_exp: Modified train_model to use Exp_POCFM.train() from exp_pocfm.py
def train_model(
    config: Config,
    stock_data: np.ndarray,
    cond_data: np.ndarray
) -> Model:
    """
    Train the PO-CFM model using Exp_POCFM.train() from exp_pocfm.py.
    
    #using_train_exp: This function now uses Exp_POCFM class instead of 
    calling train_epoch() directly. This enables proper model saving/loading,
    gradient clipping, iteration messages, and early stopping from exp file.
    
    Args:
        config: model configuration
        stock_data: [T_total, N_indicator, N_stocks]
        cond_data: [T_total, N_cond_indicator]
        
    Returns:
        trained model and training history
    """
    #using_train_exp: Import and use Exp_POCFM from exp_pocfm.py
    from exp_pocfm import Exp_POCFM
    
    #using_train_exp: Store data globally so Exp_POCFM._get_data can access it
    # This is a workaround since Exp_POCFM expects data_provider from Time-Series-Library
    global _stock_data_global, _cond_data_global, _config_global
    _stock_data_global = stock_data
    _cond_data_global = cond_data
    _config_global = config
    
    #using_train_exp: Create experiment instance with config
    exp = Exp_POCFM_Custom(config)
    
    #using_train_exp: Create setting string for checkpoint naming
    setting = f'POCFM_sl{config.seq_len}_pl{config.pred_len}_dm{config.d_model}'
    
    #using_train_exp: Call train() from exp file - this handles saving, loading,
    # gradient clipping, iteration messages, early stopping, etc.
    model = exp.train(setting)
    
    #using_train_exp: Get training history for plotting (if needed)
    train_losses = getattr(exp, 'train_losses', [])
    val_losses = getattr(exp, 'val_losses', [])
    
    #using_train_exp: Get scaler from train dataset
    train_dataset = StockDataset(
        stock_data, cond_data, 
        config.seq_len, config.pred_len,
        mode='train',
        scaler_path=config.scaler_path
    )
    
    return model, train_losses, val_losses, train_dataset.scaler


#using_train_exp: Custom Exp_POCFM class that overrides _get_data to use our StockDataset
class Exp_POCFM_Custom:
    """
    Custom experiment class that extends Exp_POCFM to use StockDataset.
    
    #using_train_exp: This class overrides _get_data() to use our custom StockDataset
    instead of data_provider from Time-Series-Library.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        #using_train_exp: Track losses for plotting
        self.train_losses = []
        self.val_losses = []
        
    def _acquire_device(self):
        #using_train_exp: Same as Exp_POCFM._acquire_device()
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        #using_train_exp: Same as Exp_POCFM._build_model()
        from POCFM import Model
        model = Model(self.args)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total number of parameters: {total_params}')
        return model
    
    def _get_data(self, flag):
        """
        #using_train_exp: Override to use StockDataset instead of data_provider.
        This allows using the custom dataset with the standard train() method.
        """
        
        #using_train_exp: Map flag to mode
        mode_map = {'train': 'train', 'val': 'val', 'test': 'test', 'pred': 'standalone'}
        mode = mode_map.get(flag, flag)
        
        #using_train_exp: Create dataset using our StockDataset
        data_set = StockDatasetWrapper(
            _stock_data_global, _cond_data_global,
            self.args.seq_len, self.args.pred_len,
            mode=mode,
            scaler_path=self.args.scaler_path
        )
        
        #using_train_exp: Create dataloader
        shuffle = (flag == 'train')
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        
        return data_set, data_loader

    def _select_optimizer(self):
        #using_train_exp: Same as Exp_POCFM._select_optimizer()
        #model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        velocity_params = set(map(id, self.model.velocity.parameters()))

        other_params = [
            p for p in self.model.parameters()
            if id(p) not in velocity_params
        ]

        model_optim = torch.optim.Adam([
            {"params": self.model.velocity.parameters(), "lr": self.args.learning_rate*2},
            {"params": other_params, "lr": self.args.learning_rate}
        ])
        return model_optim

    def _adapt_conditions(self, c):
        """
        #using_train_exp: Same as Exp_POCFM._adapt_conditions()
        Adapt condition tensor to expected dimension.
        """
        target_dim = self.args.n_cond_features
        current_dim = c.shape[-1]
        
        if current_dim < target_dim:
            padding = torch.zeros(*c.shape[:-1], target_dim - current_dim, device=c.device)
            c = torch.cat([c, padding], dim=-1)
        elif current_dim > target_dim:
            c = c[..., :target_dim]
            
        return c

    def vali(self, vali_data, vali_loader, criterion):
        """
        #using_train_exp: Same as Exp_POCFM.vali() but adapted for StockDataset format
        Validation loop.
        """
        import numpy as np
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            #using_train_exp: StockDataset returns (x_obs, x_tar, c) directly
            for i, (x_obs, x_tar, c) in enumerate(vali_loader):
                x_obs = x_obs.float().to(self.device)
                x_tar = x_tar.float().to(self.device)
                c = c.float().to(self.device)
                
                # Adapt condition dimension if needed
                if c.shape[-1] != self.args.n_cond_features:
                    c = self._adapt_conditions(c)
                
                # Compute loss
                loss, _ = self.model.compute_loss(x_obs, x_tar, c)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        #using_train_exp: Same as Exp_POCFM.train() but adapted for StockDataset format.
        This includes all features: saving/loading models, gradient clipping,
        iteration messages, learning rate scheduling, and early stopping.
        """
        import time
        import numpy as np
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping_counter = 0
        best_vali_loss = float('inf')

        model_optim = self._select_optimizer()

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            model_optim, mode='min', factor=0.5, patience=5
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            #using_train_exp: StockDataset returns (x_obs, x_tar, c) directly
            for i, (x_obs, x_tar, c) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                x_obs = x_obs.float().to(self.device)
                x_tar = x_tar.float().to(self.device)
                c = c.float().to(self.device)
                
                # Adapt condition dimension if needed
                if c.shape[-1] != self.args.n_cond_features:
                    c = self._adapt_conditions(c)
                
                # Compute flow matching loss
                loss, metrics = self.model.compute_loss(x_obs, x_tar, c)
                train_loss.append(loss.item())

                #using_train_exp: Print iteration message (from exp file)
                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                #using_train_exp: Gradient clipping (from exp file)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, None)
            test_loss = self.vali(test_data, test_loader, None)
            
            #using_train_exp: Track losses for plotting
            self.train_losses.append(train_loss_avg)
            self.val_losses.append(vali_loss)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss))
            
            # Learning rate scheduling
            scheduler.step(vali_loss)
            
            #using_train_exp: Early stopping with model saving (from exp file)
            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                early_stopping_counter = 0
                print(f'Saving model to {path}/checkpoint.pth')
                torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
                print('Saving model...')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.args.patience:
                    print('Early stopping')
                    break

        #using_train_exp: Load best model (from exp file)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


#using_train_exp: Wrapper dataset that returns data in format compatible with exp_pocfm
class StockDatasetWrapper(StockDataset):
    """
    Wrapper around StockDataset that maintains compatibility with exp_pocfm.
    
    #using_train_exp: This wrapper ensures the dataset returns data in the
    format expected by the training loop.
    """
    pass  # StockDataset already returns (x_obs, x_tar, c) which is what we need


# =============================================================================
# Visualization
# =============================================================================

def visualize_predictions(
    model: Model,
    stock_data: np.ndarray,
    cond_data: np.ndarray,
    test_dataset: StockDataset,
    config: Config,
    n_samples: int = 5,
    save_path: Optional[str] = None
):
    """Visualize model predictions vs ground truth."""
    device = torch.device(f'cuda:0' if config.use_gpu else 'cpu')
    model.eval()
    
    
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    from sklearn.preprocessing import StandardScaler
    inverse_scaler = StandardScaler()
    try:
        # Open the file in read-binary mode ('rb')
        with open(config.scaler_path, 'rb') as file:
            # Load the object from the file
            inverse_scaler = pickle.load(file)
            print("Scaler successfully loaded.")
    except Exception as e:
        print("Error loading scaler:", e)
    
    with torch.no_grad():
        for i in range(n_samples):
            idx = np.random.randint(len(test_dataset))
            x_obs, x_tar, c = test_dataset[idx]
            
            x_obs = x_obs.unsqueeze(0).to(device)
            c = c.unsqueeze(0).to(device)

            assert not x_obs[:, 0, :].sum() == x_obs[:, -1, :].sum()  # sanity check

            
            # Generate multiple predictions
            preds = []
            for _ in range(10):
                pred = model.sample(x_obs, c)
                #assert not pred[0, 0, :].sum() == pred[0, 0, :].sum()  # sanity check
                assert not pred[:, 0, :].sum() == x_obs[:, 0, :].sum()
                pred = inverse_scaler.inverse_transform(pred.squeeze(0).cpu().numpy())
                
                preds.append(pred)
            preds = np.array(preds)
            assert preds.shape[-2:] == x_obs.shape[-2:], f"Shape mismatch: {preds.shape} vs {x_obs.shape}"
             # [generated_predictions, pred_len, n_Indicator]

            #print(preds[0,:,3])

            # Renew standscaler for inverse transform
            #preds_agg = preds.mean(dim=0, keepdim=True).squeeze(0)  # aggregate predictions
            #print(f'Aggregated predictions shape: {preds_agg.shape}')
            #full_x = torch.cat([x_obs, preds_agg], dim=-2) # [batch, seq_len + pred_len, n_Indicator]
            #full_x = full_x.cpu().numpy()
            #print('full_x shape:', full_x.shape)
            

            #inverse_scaler.fit(full_x.reshape(-1, full_x.shape[-1]))  
            #inverse_scaler.mean_ = x_obs[:, -1,:].squeeze(0).cpu().numpy()  # Anchor to last observed value

            #print('inverse_scaler.mean_ shape:', inverse_scaler.mean_.shape)
            #print('inverse_scaler.scale_ shape:', inverse_scaler.scale_.shape)
            #print(f'inverse_scaler.mean_: {inverse_scaler.mean_}')
            #print(f'inverse_scaler.scale_: {inverse_scaler.scale_}')

            
            # Plot close price
            feature_idx = 2
            ax = axes[i]

            
            
            
            # Observed
            #print(x_obs.squeeze(1)[0,:,:].squeeze(0).cpu().numpy())
            reversed_x_obs = inverse_scaler.inverse_transform(x_obs.squeeze(1)[0,:,:].squeeze(0).cpu().numpy())
            #print(f'reversed_x_obs[:, feature_idx]: {reversed_x_obs[:, feature_idx]}')
            t_obs = np.arange(config.seq_len)
            ax.plot(t_obs, reversed_x_obs[:, feature_idx], 
                   'b-', label='Observed', linewidth=2)
            
            # Target
            reversed_x_tar = inverse_scaler.inverse_transform(x_tar.cpu().numpy())
            t_tar = np.arange(config.seq_len, config.seq_len + config.pred_len)
            ax.plot(t_tar, reversed_x_tar[:, feature_idx], 
                   'g-', label='Ground Truth', linewidth=2)
            
            # Predictions
            #print(f'preds[0].squeeze(0).cpu().numpy() before inverse transform: {preds[0].squeeze(0).cpu().numpy().shape}')
            #reversed_preds = inverse_scaler.inverse_transform(preds[0]) # collapse generated_predictions and batch
            #print(f'reversed_preds shape: {reversed_preds.shape}')
            pred_mean = preds.mean(axis=0)[:, feature_idx]
            pred_std = preds.std(axis=0)[:, feature_idx]

            ax.set_xlim(-5, config.seq_len+config.pred_len)

            ax.plot(t_tar, pred_mean, 'r--', label='Prediction (mean)', linewidth=2)
            #ax.fill_between(t_tar, pred_mean - 2*pred_std, pred_mean + 2*pred_std,
            #              alpha=0.3, color='red', label='±2 std')
            
            ax.axvline(x=config.seq_len - 0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value of column index ' + str(feature_idx))
            ax.set_title(f'Sample {i + 1}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    

def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Partially Observable Conditional Flow Matching (PO-CFM)")
    print("=" * 60)

    # Configuration
    #using_train_exp: Added patience and checkpoints to config
    config = Config(
        seq_len=60,
        pred_len=60,
        n_features=6,          # Stock indicators
        n_cond_features=10,    # Market conditions
        d_model=128,
        n_heads=4,
        e_layers=3,
        d_layers=2,
        d_ff=256,
        dropout=0.1,
        n_steps=200,
        learning_rate=1e-5,
        batch_size=128,
        train_epochs=100,
        use_gpu=True,
        #using_train_exp: Set patience for early stopping
        patience=10,
        #using_train_exp: Set checkpoints directory
        checkpoints='./checkpoints/'
    )
    
    print("\n1. Loading data...")
    # stock_data, cond_data = generate_dummy_stock_data(
    #     T_total=1000,
    #     N_stocks=20,
    #     N_indicator=config.enc_in,
    #     N_cond_indicator=config.n_cond_features,
    #     noise_level=0.1
    # )

    stock_data = np.float32(np.load('2020-01-01-2025-05-31-637.npy').astype(np.float32))  # [T, N_indicator, N_stocks]
    print(f"   Stock data shape: {stock_data.shape}")
    cond_data = pd.read_csv('metamarket-20200101-20250531.csv').to_numpy().astype(np.float32)
    print(f"   Condition data shape: {cond_data.shape}")
    

    
    isnan = np.isnan(stock_data).any() or np.isinf(stock_data).any() or np.isnan(cond_data).any() or np.isinf(cond_data).any()
    if isnan:
        raise ValueError(f"    Input data contains NaN or Inf values.")
    else:
        print(f"   Input data contains no NaN or Inf values.")

    #using_train_exp: Ensure save directories exist
    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.checkpoints, exist_ok=True)

    # Train model
    #using_train_exp: train_model now uses Exp_POCFM_Custom.train() internally
    print("\n2. Training PO-CFM model...")

    # temp test of limiting stocks
    # ticker_index_map = pickle.load(open('ticker_index_dict.pkl', 'rb'))
    # inverted_dict = {v: k for k, v in ticker_index_map.items()}
    # selected_stock_ticker = ['2603', '2359', '2408', '1301', '1513', '3037', '3450', '8210', '2382', '1513', '1519']
    # selected_stock_idx = [inverted_dict[ticker] for ticker in selected_stock_ticker]
    # selected_stock_data = stock_data[:, :, selected_stock_idx]
    
    # model, train_losses, val_losses, scaler = train_model(
    #     config, stock_data, cond_data
    # )
    # device = next(model.parameters()).device

    model, train_losses, val_losses, scaler = train_model(
        config, stock_data, cond_data
    )
    device = next(model.parameters()).device

    # Evaluate on test set
    print("\n3. Evaluating on test set...")
    device = torch.device(f'cuda:0' if config.use_gpu else 'cpu')
    
    test_dataset = StockDataset(
        stock_data, cond_data,
        config.seq_len, config.pred_len,
        mode='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    _, test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"   Test MSE: {test_mse:.6f}")
    print(f"   Test MAE: {test_mae:.6f}")
    
    # Visualize
    print("\n4. Visualizing losses history...")
    #using_train_exp: Plot training curves from exp training
    if train_losses and val_losses:
        plot_training_curves(train_losses, val_losses, 
                             save_path=config.save_path + 'training_curves.png')
    
    
    # visualize_predictions(model, stock_data, cond_data, config, n_samples=20, save_path=config.save_path + 'pocfm_stock_predictions.png')

    
    
    # standalone test on selected stocks
    print("\n5. Evaluating on selected voltile stocks...")

    ticker_index_map = pickle.load(open('ticker_index_dict.pkl', 'rb'))
    inverted_dict = {v: k for k, v in ticker_index_map.items()}
    # selected_stock_ticker = ['1513', '1519']
    # selected_stock_idx = [inverted_dict[ticker] for ticker in selected_stock_ticker]
    # selected_stock_data = stock_data[:, :, selected_stock_idx]

    # test_dataset = StockDataset(
    #     selected_stock_data, cond_data,
    #     config.seq_len, config.pred_len,
    #     mode='standalone'
    # )
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    # visualize_predictions(model, selected_stock_data, cond_data,test_dataset, config, n_samples=20, save_path=config.save_path + 'pocfm_selected_stock_predictions.png')
    # _, test_mse, test_mae = evaluate(model, test_loader, device)
    # print(f"   Test MSE: {test_mse:.6f}")
    # print(f"   Test MAE: {test_mae:.6f}")

    print("\n6. Evaluating on out-of-time windows...")

    outtime_stock_data = np.float32(np.load('2025-06-01-2025-12-31-637.npy').astype(np.float32))  # [T, N_indicator, N_stocks]
    print(f"   Stock data shape: {outtime_stock_data.shape}")
    cond_data = pd.read_csv('metamarket-20250601-20251231.csv').to_numpy().astype(np.float32)
    print(f"   Condition data shape: {cond_data.shape}")

    test_dataset = StockDataset(
        outtime_stock_data, cond_data,
        config.seq_len, config.pred_len,
        mode='standalone'
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    visualize_predictions(model, outtime_stock_data, cond_data,test_dataset, config, n_samples=20, save_path=config.save_path + 'pocfm_outtime_stock_predictions.png')
    _, test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"   Test MSE: {test_mse:.6f}")
    print(f"   Test MAE: {test_mae:.6f}")

    print("\n7. Evaluating on interested all stocks...")
    interested_stock_ticker = ['1513','1519']
    interested_stock_idx = [inverted_dict[ticker] for ticker in interested_stock_ticker]
    interested_stock_data = outtime_stock_data[:, :, interested_stock_idx] 
    
    test_dataset = StockDataset(
        interested_stock_data, cond_data,
        config.seq_len, config.pred_len,
        mode='standalone'
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    visualize_predictions(model, interested_stock_data, cond_data,test_dataset, config, n_samples=20, save_path=config.save_path + 'pocfm_interested_stock_predictions.png')
    _, test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"   Test MSE: {test_mse:.6f}")
    print(f"   Test MAE: {test_mae:.6f}")

    
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)




if __name__ == "__main__":
    main()
