"""
Experiment class for Partially Observable Conditional Flow Matching (PO-CFM).

This extends the standard long-term forecasting experiment to use flow matching loss
instead of standard MSE loss.
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError
from utils.tools import visual

warnings.filterwarnings('ignore')


class Exp_POCFM:
    """Experiment class for PO-CFM model training and evaluation."""
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _acquire_device(self):
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
        from POCFM import Model
        model = Model(self.args)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        """
        Get data loader for training, validation, or testing.
        
        This uses a custom dataset that provides:
        - x_obs: observed sequence
        - x_tar: target sequence to predict
        - c: condition sequence (external features)
        """
        from data_provider.data_factory import data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # Flow matching uses its own loss computation
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        """Validation loop."""
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare inputs
                x_obs = batch_x  # [B, seq_len, n_features]
                x_tar = batch_y[:, -self.args.pred_len:, :]  # [B, pred_len, n_features]
                c = batch_x_mark  # Use time features as conditions
                
                # Adapt condition dimension if needed
                if c.shape[-1] != self.args.n_cond_features:
                    c = self._adapt_conditions(c)
                
                # Compute loss
                loss, _ = self.model.compute_loss(x_obs, x_tar, c)
                total_loss.append(loss.item())
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def _adapt_conditions(self, c):
        """Adapt condition tensor to expected dimension."""
        target_dim = self.args.n_cond_features
        current_dim = c.shape[-1]
        
        if current_dim < target_dim:
            # Pad with zeros
            padding = torch.zeros(*c.shape[:-1], target_dim - current_dim, device=c.device)
            c = torch.cat([c, padding], dim=-1)
        elif current_dim > target_dim:
            # Truncate
            c = c[..., :target_dim]
            
        return c

    def train(self, setting):
        """Training loop."""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.save_path, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping_counter = 0
        best_vali_loss = float('inf')

        model_optim = self._select_optimizer()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            model_optim, mode='min', factor=0.5, patience=3
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare inputs
                x_obs = batch_x  # [B, seq_len, n_features]
                x_tar = batch_y[:, -self.args.pred_len:, :]  # [B, pred_len, n_features]
                c = batch_x_mark  # Use time features as conditions
                
                # Adapt condition dimension if needed
                if c.shape[-1] != self.args.n_cond_features:
                    c = self._adapt_conditions(c)
                
                # Compute flow matching loss
                loss, metrics = self.model.compute_loss(x_obs, x_tar, c)
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, None)
            test_loss = self.vali(test_data, test_loader, None)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Learning rate scheduling
            scheduler.step(vali_loss)
            
            # Early stopping
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

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """Test loop with MSE and MAE metrics."""
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                x_obs = batch_x
                x_tar = batch_y[:, -self.args.pred_len:, :]
                c = batch_x_mark
                
                if c.shape[-1] != self.args.n_cond_features:
                    c = self._adapt_conditions(c)
                
                # Generate predictions
                outputs = self.model.sample(x_obs, c)
                
                pred = outputs.detach().cpu().numpy()
                true = x_tar.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # Compute metrics
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        rmse = np.sqrt(mse)
        
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        
        # Save results
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mse, mae

    def predict(self, setting, load=False):
        """Generate predictions for the entire test set."""
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                x_obs = batch_x
                c = batch_x_mark
                
                if c.shape[-1] != self.args.n_cond_features:
                    c = self._adapt_conditions(c)
                
                outputs = self.model.sample(x_obs, c)
                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return preds
