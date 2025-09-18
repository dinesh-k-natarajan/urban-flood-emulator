from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

# LightningDataModule class that configures dataloaders using train/val/test Dataset objects
class DataModule(LightningDataModule):
    def __init__(
            self, 
            train_dataset: Dataset,
            val_dataset: Dataset,
            test_dataset: Dataset,
            batch_size: int = 64,
            num_workers: int = 0
        ) -> None:
        super(DataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.test_dataset  = test_dataset
        self.batch_size    = batch_size
        self.num_workers   = num_workers
    
    def train_dataloader(self) -> DataLoader: 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS: 
        pass