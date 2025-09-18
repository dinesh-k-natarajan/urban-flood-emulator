from typing import Optional, Union, Dict, Tuple
import torch
import torchmetrics
from torch.nn import Module
import pytorch_lightning as pl

# Model class defining how to configure the model for training and inference
class RegressionModel(pl.LightningModule):
    def __init__(
            self, 
            architecture: Module, 
            criterion, 
            optimizer: torch.optim.Optimizer,
            metric_collection: torchmetrics.MetricCollection,
        ) -> None:
        super(RegressionModel, self).__init__()
        self.save_hyperparameters(ignore=["architecture", "metric_collection"])
        
        self.model = architecture
        self.criterion = criterion
        self.optimizer = optimizer
        # Other metrics to be tracked on train, val and test sets
        self.train_metrics = metric_collection.clone(prefix="train_")
        self.val_metrics   = metric_collection.clone(prefix='val_')
        self.test_metrics  = metric_collection.clone(prefix='test_')
        # Lists for storing and computing losses
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(**x)
        loss = self.criterion(y_hat['water_depth'], y['water_depth'])
        self.training_step_outputs.append(loss.item())
        self.train_metrics.update(y_hat['water_depth'], y['water_depth'])
        return loss
    
    def on_train_epoch_end(self) -> None:
        # Compute loss
        loss_per_epoch = torch.Tensor(self.training_step_outputs).mean()
        print(f'After epoch {self.current_epoch}, train_loss = {loss_per_epoch}')
        self.log('train_loss', loss_per_epoch)
        self.training_step_outputs.clear() # free memory
        # Compute metrics 
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(**x)
        val_loss = self.criterion(y_hat['water_depth'], y['water_depth'])
        self.validation_step_outputs.append(val_loss.item())
        self.val_metrics.update(y_hat['water_depth'], y['water_depth'])
        return val_loss
    
    def on_validation_epoch_end(self) -> None:
        # Compute loss
        loss_per_epoch = torch.Tensor(self.validation_step_outputs).mean()
        print(f'After epoch {self.current_epoch}, val_loss = {loss_per_epoch}')
        self.log('val_loss', loss_per_epoch)
        self.validation_step_outputs.clear() # free memory
        # Compute metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(**x)
        test_loss = self.criterion(y_hat['water_depth'], y['water_depth'])
        self.test_step_outputs.append(test_loss.item())
        self.test_metrics.update(y_hat['water_depth'], y['water_depth'])
        return test_loss
    
    def on_test_epoch_end(self) -> None:
        # Compute loss
        loss_per_epoch = torch.Tensor(self.test_step_outputs).mean()
        self.log('test_loss', loss_per_epoch)
        self.test_step_outputs.clear() # free memory
        # Compute metrics
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()
    
    def predict_step(self, batch, batch_idx=None):
        """Predit step of the regression model returns the 
        outputs of the model.
        """
        x, y = batch
        return self.model(**x)
    