import yaml
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from os.path import join

import utils
import models

import torch
import pandas as pd
import pytorch_lightning as pl
import torchmetrics.regression as tmreg
from torch.nn import MSELoss
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Parse arguments
parser = argparse.ArgumentParser()
## This script requires the name of the config file as command line argument
parser.add_argument("-c", "--Config", help="Name of config file")
args = parser.parse_args()

# Load configs from config file into a configs dictionary
config_path = args.Config
with open(config_path, 'r') as f:
    configs = yaml.safe_load(f)
    
# Extract necessary configs
dataset_name = configs['DATASET_NAME']
BATCH_SIZE = configs['BATCH_SIZE']
NUM_WORKERS = configs['NUM_WORKERS']
arch_name = configs['ARCH_NAME']
fusion_name = configs['FUSION_NAME']
EPOCHS = configs['EPOCHS']
LEARNING_RATE = configs['LEARNING_RATE']
OPTIMIZER = configs['OPTIMIZER']
model_name = configs['MODEL_NAME']
NUM_GPUS = configs['NUM_GPUS']
PRECISION = configs['PRECISION']
ACC_GRAD = configs['ACC_GRAD']

# Define paths to dataset
dataset_dir = "./data/" # TODO: define the path to data directory
city_dir = join(dataset_dir, "city_level_data")
rainfall_dir = join(dataset_dir, "rainfall_events")

# Load training, validation and test city names
sets_file = join(dataset_dir, "sets.csv")
sets_df = pd.read_csv(sets_file)
train_cities = sets_df[sets_df["set"] == "train"]["City"].to_list()
val_cities = sets_df[sets_df["set"] == "val"]["City"].to_list()
test_cities = sets_df[sets_df["set"] == "test"]["City"].to_list()

# Choose appropriate Dataset class based on config
dataset_mod = getattr(utils, dataset_name)
print(100*'-')
print(f"Using the {dataset_name} dataset...")

# Set up train/val/test dataset objects
train_dataset = dataset_mod("train", cities_list=train_cities, city_dir=city_dir, rainfall_dir=rainfall_dir)
print(f"Number of data samples in training set = {len(train_dataset)}")
print(100*'-')
val_dataset = dataset_mod("validation", cities_list=val_cities, city_dir=city_dir, rainfall_dir=rainfall_dir)
print(f"Number of data samples in validation set = {len(val_dataset)}")
print(100*'-')
test_dataset = dataset_mod("test", cities_list=test_cities, city_dir=city_dir, rainfall_dir=rainfall_dir)
print(f"Number of data samples in test set = {len(test_dataset)}")
print(100*'-')

# Get number of inputs channels in the dataset
n_input_channels = train_dataset.get_input_channels()

# Setup the DataModule using the three datasets
datamodule = utils.DataModule(
    train_dataset=train_dataset, 
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS,
)

# Setup the architecture
## Load the architecture module based on the arch name
arch_module = getattr(models, arch_name)
## Initialize the architecture class
arch = arch_module(
    n_input_channels=n_input_channels,
    fusion_module=fusion_name,
)

# Optimization configurations
optimizer_mod = getattr(torch.optim, OPTIMIZER)
optimizer = optimizer_mod(arch.parameters(), lr=LEARNING_RATE)
# Loss function
criterion = MSELoss()

# Other metrics to track
rmse = tmreg.MeanSquaredError(squared=False)
csi_5cm = tmreg.CriticalSuccessIndex(threshold=0.05)
csi_25cm = tmreg.CriticalSuccessIndex(threshold=0.25)
pixelwise_r2 = utils.PixelwiseCoefficientOfDetermination()
metrics_dict = {
    'RMSE': rmse, 
    'CSI_5cm': csi_5cm,
    'CSI_25cm': csi_25cm,
    'pixelwise_r2': pixelwise_r2,
}
metric_collection = MetricCollection(metrics_dict)
 
# Setup the model
## Load the model module based on the model name
model_module = getattr(models, model_name)
model = model_module(
    architecture=arch,
    criterion=criterion,
    optimizer=optimizer,
    metric_collection=metric_collection,
)
print(100*'-')
print(f"Model has been built:\n{model.model}")

# Define the callbacks during training
checkpoint_cb = ModelCheckpoint(
    save_last=True,
    save_top_k=2,
    monitor='val_loss',
    filename='{epoch}_{val_loss:.16f}',
)
earlystopping_cb = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    verbose= False,
    patience= EPOCHS//3, # no. of non-improving epochs to wait before stopping training
    check_finite= True, # stops training if NaN is encountered in loss
)
callbacks_list = [
    checkpoint_cb,
    earlystopping_cb,
]

# Setup the logger
logger = pl.loggers.CSVLogger(
    save_dir = "./exp/logs", 
)

# Setup the trainer
trainer = pl.Trainer(
    accumulate_grad_batches=ACC_GRAD,
    max_epochs=EPOCHS,
    accelerator='gpu', devices=NUM_GPUS,
    logger=logger,
    callbacks=callbacks_list,
    log_every_n_steps=100,
    enable_progress_bar=False,
    num_sanity_val_steps=0,
    precision=PRECISION,
)

# Train the model
print(100*'-')
print("Training the model...")
start_time = time.time()
trainer.fit(
    model=model, 
    train_dataloaders=datamodule.train_dataloader(), 
    val_dataloaders=datamodule.val_dataloader()
)
training_time = time.time() - start_time
print(f'Time taken for training = {training_time} seconds...')

# Evaluate model on test set
print(100*'-')
print("Evaluating the trained model on test set...")
test_metrics = trainer.test(
    model=model,
    ckpt_path='best',
    dataloaders=datamodule.test_dataloader(),
)
