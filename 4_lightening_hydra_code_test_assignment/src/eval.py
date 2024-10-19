import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)

def instantiate_callbacks(callback_cfg: DictConfig) -> List[L.Callback]:
    callbacks: List[L.Callback] = []
    if not callback_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callback_cfg.items():
        if "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    loggers: List[Logger] = []
    if not logger_cfg:
        log.warning("No logger configs found! Skipping..")
        return loggers

    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers

import os
import glob

def get_latest_checkpoint(base_dir):
    base_dir = Path(base_dir)
    print(f"Looking for checkpoints starting from directory: {base_dir}")

    # Start from the base_dir and search upwards until we find a checkpoint
    current_dir = base_dir
    while current_dir != current_dir.parent:  # Stop when we reach the root directory
        checkpoint_pattern = str(current_dir / "**" / "checkpoints" / "*.ckpt")
        print(f"Searching with pattern: {checkpoint_pattern}")
        
        checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
        if checkpoint_files:
            print(f"Checkpoint files found: {checkpoint_files}")
            return max(checkpoint_files, key=os.path.getctime)
        
        current_dir = current_dir.parent

    raise FileNotFoundError(f"No checkpoints found in or above {base_dir}")


@task_wrapper
def evaluate(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting evaluation!")
    # Ensure the datamodule is set up
    datamodule.setup(stage="test")

    # Get the test dataloader
    test_loader = datamodule.test_dataloader()

    base_dir = cfg.paths.output_dir
    try:
        ckpt_path = get_latest_checkpoint(base_dir)
        log.info(f"Using checkpoint: {ckpt_path}")    
        log.info(f"Loading checkpoint: {ckpt_path}")
        test_metrics = trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path)
    except FileNotFoundError:
        log.warning("No checkpoint found. Using current model weights.")
        test_metrics = trainer.test(model=model, dataloaders=test_loader)
    log.info(f"Test metrics:\n{test_metrics}")
    return test_metrics
  

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg=cfg))

    # Set up paths
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "eval_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Set up callbacks
    callbacks: List[L.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Set up loggers
    loggers: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Evaluate the model
    evaluate(cfg, trainer, model, datamodule)

if __name__ == "__main__":
    main()