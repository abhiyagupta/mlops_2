
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lightning as L
from PIL import Image
from torchvision import transforms
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import random

from src.models.dogs_classifier import DogsBreedClassifier
from src.utils.logging_utils import setup_logger, task_wrapper, get_rich_progress
from rich.console import Console
import rootutils
import os
import glob

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)


@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return img, transform(img).unsqueeze(0)


@task_wrapper
def infer(model,image_tensor):
    class_labels= ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever','Labrador_Retriever', 'Poodle','Rottweiler','Yorkshire_Terrier']
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor) 
        probabilities = F.softmax(logits,dim=-1)
        y_pred = probabilities.argmax(dim=-1).item()

    confidence = probabilities[0][y_pred].item()
    predicted_label = class_labels[y_pred]
    return predicted_label,confidence

def instantiate_callbacks(callback_cfg):
    callbacks = []
    if callback_cfg:
        for _, cb_conf in callback_cfg.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

def instantiate_loggers(logger_cfg):
    loggers = []
    if logger_cfg:
        for _, lg_conf in logger_cfg.items():
            if "_target_" in lg_conf:
                loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

@task_wrapper
def evaluate(trainer, model, datamodule, ckpt_path):
    if ckpt_path:
        # Use the test_dataloader from the datamodule
        test_dataloader = datamodule.test_dataloader()
        trainer.test(model, test_dataloader, ckpt_path=ckpt_path)
    else:
        logging.error("No checkpoint path provided. Cannot evaluate the model.")
        return
    logging.info(f"Evaluation metrics: {trainer.callback_metrics}")




@task_wrapper
def save_and_display_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    
    # Save the image to the output path
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    
    # Display the image in the terminal
    #plt.show()  # Add this line to display the image inline
    plt.close()


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



@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    # Set up paths
    log_dir = Path(cfg.paths.log_dir)
    setup_logger(log_dir / "infer_log.log")

    base_dir = cfg.paths.output_dir
    ckpt_path = get_latest_checkpoint(base_dir)
    log.info(f"Using checkpoint: {ckpt_path}")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogsBreedClassifier.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    input_folder = Path(cfg.input_path)
    output_folder = Path(cfg.output_path)
    output_folder.mkdir(exist_ok=True, parents=True)

    print(f"input_folder: {input_folder}")
    print(f"output_folder: {output_folder}")

    # Get all image files from the validation folder
    image_files = list(input_folder.glob("**/*.jpg")) + list(input_folder.glob("**/*.png"))
    
    # Randomly select 5 images (or less if there are fewer than 5 images)
    selected_images = random.sample(image_files, min(cfg.num_images, len(image_files)))
    print (len(selected_images))
    console = Console()
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(selected_images))

        for image_file in selected_images:
            img, img_tensor = load_image(image_file)
            predicted_label, confidence = infer(model, img_tensor.to(device))

            output_file = output_folder / f"{image_file.stem}_prediction.png"
            save_and_display_prediction_image(img, predicted_label, confidence, output_file)
            print(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
            log.info(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
            progress.advance(task) 
    log.info(f"Predictions saved to: {output_folder}")

if __name__ == "__main__":
    main()