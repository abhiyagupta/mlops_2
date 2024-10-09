import os
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import logging

log = logging.getLogger(__name__)


class DogsBreedDataModule(pl.LightningDataModule):
    def __init__(
                self,
                batch_size:int,
                num_workers:int,
                pin_memory:bool,
                seed: Optional[int] = 42,
                data_dir: Optional[str] = "dogs_dataset",  # Default data directory
                # data_dir:Optional[AnyStr]=None,

    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Initialize _dl_path here
        self._dl_path = Path(data_dir) 
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None        
        #self.prepare_data()s

    def prepare_data(self):
        """Download images and prepare datasets."""
        dataset_dir = self._dl_path.joinpath("dataset")
        log.info(f"Checking for dataset in: {dataset_dir}")
        
        print(f"Checking for dataset in: {dataset_dir}")

        if not dataset_dir.exists():
            log.info("Dataset not found. Downloading...")
            download_and_extract_archive(
                url="https://raw.githubusercontent.com/abhiyagupta/Datasets/refs/heads/main/CNN_Datasets/dogs_classifier_dataset.zip",
                download_root=self._dl_path,
                remove_finished=True
            )
        else:
            log.info("Dataset already exists.")

        # Check if the dataset has the expected structure
        if not any(dataset_dir.glob("*/*.jpg")):
            raise RuntimeError(f"No images found in {dataset_dir}. The dataset may be corrupted or have an unexpected structure.")
    


    def setup(self, stage: Optional[str] = None):

      """Load data and split into train, val, test sets."""
      print("Setting up data...")

      if self.train_dataset is not None:
        return  # Data is already setup      

      self.prepare_data()  # Ensure data is downloaded and extracted

      data_images = self.create_dataframe()  # Only train/test images here
      print(f"Total images found for train-test split: {len(data_images)}")  # Excluding validation images

      self.num_classes = data_images['label'].nunique()
      print(f"Number of unique classes: {self.num_classes}")

      if len(data_images) == 0 or self.num_classes == 0:
        raise RuntimeError("No images found or no unique classes. Check the dataset structure and content.")      

      train_df, test_df = self.split_train_test(data_images)

      print(f"Train set size: {len(train_df)}")
      print(f"Test set size: {len(test_df)}")

      self.train_dataset = self.create_dataset(train_df)
      self.test_dataset = self.create_dataset(test_df)

      # Handle validation data separately
      val_dir = self._dl_path.joinpath("dataset", "validation")
      if not val_dir.exists():
          val_dir.mkdir(parents=True, exist_ok=True)
          self.prepare_validation_data(train_df)
    
      val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
      
      self.val_dataset = ImageFolder(root=str(val_dir), transform=val_transform)
      print(f"Number of validation images: {len(self.val_dataset)}")


    #   # Print validation images number 
    #   num_val_images = len(list(val_dir.glob("*/*.jpg")))
    #   print(f"Number of validation images: {num_val_images} (not included in the train-test split)")

    #   # Move validation images to the input folder
    #   input_folder = Path("input")
    #   if not input_folder.exists():
    #       input_folder.mkdir(parents=True, exist_ok=True)

    #   for img_path in val_dir.glob("*/*.jpg"):
    #       dst_path = input_folder.joinpath(img_path.name)
    #       shutil.copy(img_path, dst_path)

    #   print(f"Validation images saved to: {input_folder.absolute()}")


            

    def create_dataframe(self):
        DATASET_PATH = self._dl_path.joinpath("dataset")
        
        print(f"Looking for images in: {DATASET_PATH}")
        IMAGE_PATH_LIST = list(DATASET_PATH.glob("*/*.jpg"))
       
        IMAGE_PATH_LIST = [path for path in IMAGE_PATH_LIST if "validation" not in str(path)]
        print(f"Number of images found (excluding validation): {len(IMAGE_PATH_LIST)}")
        images_path = [str(img_path.relative_to(DATASET_PATH)) for img_path in IMAGE_PATH_LIST]
        labels = [img_path.parent.name for img_path in IMAGE_PATH_LIST]

        df = pd.DataFrame({'image_path': images_path, 'label': labels})
        print(f"Number of unique labels: {df['label'].nunique()}")
        print(f"dataframe  is: {df.head()}")
        print(f"shape of dataframe is {df.shape}") 
        return df

    def split_train_test(self, df):
        train_split_idx, test_split_idx, _, _ = train_test_split(
            df.index, df.label, test_size=0.2, stratify=df.label, random_state=self.hparams.seed
        )
        return df.iloc[train_split_idx].reset_index(drop=True), df.iloc[test_split_idx].reset_index(drop=True)

    def prepare_validation_data(self, train_df):
        val_split_idx, _, _, _ = train_test_split(
            train_df.index, train_df.label, test_size=0.8, stratify=train_df.label, random_state=self.hparams.seed
        )
        val_df = train_df.iloc[val_split_idx].reset_index(drop=True)
        
        for _, row in val_df.iterrows():
            src_path = self._dl_path.joinpath("dataset", row['image_path'])
            dst_path = self._dl_path.joinpath("dataset", "validation", row['label'], src_path.name)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src_path, dst_path)

    def create_dataset(self, df):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return CustomImageDataset(
            root=str(self._dl_path.joinpath("dataset")),
            image_paths=df['image_path'],
            transform=transform
        )

    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=True)

    def train_dataloader(self) -> DataLoader:
        self.setup()
        return DataLoader(
                    dataset=self.train_dataset,
                    batch_size=self.hparams.batch_size,
                    shuffle=True,
                    pin_memory=self.hparams.pin_memory,
                    num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        self.setup()
        # val_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # val_dataset = ImageFolder(root=str(self._dl_path.joinpath("dataset", "validation")), transform=val_transform)
        return DataLoader(
                    dataset=self.val_dataset, 
                    batch_size=self.hparams.batch_size,
                    shuffle=False,
                    pin_memory=self.hparams.pin_memory,
                    num_workers=self.hparams.num_workers
        )

  

    def test_dataloader(self) -> DataLoader:
      self.setup()
      return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                pin_memory=self.hparams.pin_memory,
                num_workers=self.hparams.num_workers
    )

class CustomImageDataset(Dataset):
    def __init__(self, root, image_paths, transform=None):
        self.root = root
        self.image_paths = image_paths
        self.transform = transform
        self.images = []
        self.labels = []

        # for idx, path in enumerate(image_paths):
        #     img = Image.open(os.path.join(root, path))
        #     self.images.append(img)
        #     self.labels.append(idx)  # Assuming labels are indices
        # Create a mapping of unique labels to integers
        unique_labels = sorted(set(path.split('/')[0] for path in image_paths))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        for path in image_paths:
            img = Image.open(os.path.join(root, path))
            self.images.append(img)
            label = path.split('/')[0]  # Assuming the first part of the path is the label
            self.labels.append(self.label_to_idx[label])
        
        self.num_classes = len(unique_labels)
        print(f"Number of classes in CustomImageDataset: {self.num_classes}")        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label






# =============================================
# # output of this cell in colab:
# Checking for dataset in: dogs_dataset/dataset
# Checking for dataset in: dogs_dataset/dataset
# Setting up data...
# Looking for images in: dogs_dataset/dataset
# Number of images found (excluding validation): 813
# Number of unique labels: 10
# dataframe  is:                                    image_path              label
# 0   Yorkshire_Terrier/Yorkshire Terrier_8.jpg  Yorkshire_Terrier
# 1   Yorkshire_Terrier/Yorkshire Terrier_7.jpg  Yorkshire_Terrier
# 2  Yorkshire_Terrier/Yorkshire Terrier_25.jpg  Yorkshire_Terrier
# 3   Yorkshire_Terrier/Yorkshire Terrier_9.jpg  Yorkshire_Terrier
# 4  Yorkshire_Terrier/Yorkshire Terrier_73.jpg  Yorkshire_Terrier
# shape of dataframe is (813, 2)
# Total images found for train-test split: 813
# Number of unique classes: 10
# Train set size: 650
# Test set size: 163
# Number of validation images: 154 (not included in the train-test split)
# Validation images saved to: /content/input

