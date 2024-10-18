import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
import torch
import matplotlib.pyplot as plt
import glob
import random
import torch
import torch.nn.functional as F

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.infer import load_image, infer, save_and_display_prediction_image, get_latest_checkpoint, main
from src.models.dogs_classifier import DogsBreedClassifier

@pytest.fixture(scope="function")
def config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="infer")
    return cfg

@patch("src.infer.Image.open")
@patch("src.infer.transforms.Compose")
def test_load_image(mock_compose, mock_image_open, config):
    mock_image = MagicMock()
    mock_image_open.return_value.convert.return_value = mock_image
    mock_transform = MagicMock()
    mock_compose.return_value = mock_transform

    # Get a random image path from input_path
    image_paths = glob.glob(str(Path(config.input_path) / "*.jpg"))
    image_path = random.choice(image_paths) if image_paths else "dummy_path.jpg"

    img, img_tensor = load_image(image_path)

    assert img == mock_image
    mock_image_open.assert_called_once_with(image_path)
    mock_transform.assert_called_once()

@patch("torch.no_grad")
def test_infer(mock_no_grad):
    mock_model = MagicMock()
    mock_image_tensor = torch.rand(1, 3, 224, 224)  # Create a random tensor

    # Create a mock logits tensor
    mock_logits = torch.tensor([[0.1, 0.2, 0.7]])
    mock_model.return_value = mock_logits

    predicted_label, confidence = infer(mock_model, mock_image_tensor)

    assert isinstance(predicted_label, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1  # Confidence should be between 0 and 1
    mock_model.assert_called_once_with(mock_image_tensor)

    # Check if the predicted label corresponds to the highest logit
    expected_label_index = mock_logits.argmax(dim=-1).item()
    expected_label = ['Beagle', 'Boxer', 'Bulldog', 'Dachshund', 'German_Shepherd', 'Golden_Retriever',
                      'Labrador_Retriever', 'Poodle', 'Rottweiler', 'Yorkshire_Terrier'][expected_label_index]
    assert predicted_label == expected_label

    # Check if the confidence matches the softmax of the highest logit
    expected_confidence = F.softmax(mock_logits, dim=-1)[0, expected_label_index].item()
    assert abs(confidence - expected_confidence) < 1e-6  # Allow for small floating-point differences

@pytest.fixture
def mock_hydra_config():
    return OmegaConf.create({
        "paths": {
            "output_dir": "/mock/output/dir",
            "log_dir": "/mock/log/dir"
        },
        "input_path": "/mock/input/path",
        "output_path": "/mock/output/path",
        "num_images": 5
    })

@patch("matplotlib.pyplot.figure")
@patch("matplotlib.pyplot.imshow")
@patch("matplotlib.pyplot.axis")
@patch("matplotlib.pyplot.title")
@patch("matplotlib.pyplot.tight_layout")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
@patch("hydra.core.hydra_config.HydraConfig.get")
def test_save_and_display_prediction_image(mock_hydra_get, mock_close, mock_savefig, mock_tight_layout, mock_title, mock_axis, mock_imshow, mock_figure, mock_hydra_config):
    mock_hydra_get.return_value = mock_hydra_config
    
    image = MagicMock()
    predicted_label = "Beagle"
    confidence = 0.95
    output_path = Path("/mock/output/path/test_output.png")

    save_and_display_prediction_image(image, predicted_label, confidence, output_path)

    mock_figure.assert_called_once_with(figsize=(5, 5))
    mock_imshow.assert_called_once_with(image)
    mock_axis.assert_called_once_with("off")
    mock_title.assert_called_once_with(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    mock_tight_layout.assert_called_once()
    mock_savefig.assert_called_once_with(output_path, dpi=300, bbox_inches="tight")
    mock_close.assert_called_once()

@patch("glob.glob")
@patch("os.path.getctime")
@patch("hydra.core.hydra_config.HydraConfig.get")
def test_get_latest_checkpoint(mock_hydra_get, mock_getctime, mock_glob, mock_hydra_config):
    mock_hydra_get.return_value = mock_hydra_config
    base_dir = Path(mock_hydra_config.paths.output_dir)
    mock_glob.return_value = [str(base_dir / "checkpoints" / "checkpoint1.ckpt"), 
                              str(base_dir / "checkpoints" / "checkpoint2.ckpt")]
    mock_getctime.side_effect = [1000, 2000]

    result = get_latest_checkpoint(base_dir)

    assert result == str(base_dir / "checkpoints" / "checkpoint2.ckpt")


@patch("src.infer.get_latest_checkpoint")
@patch("src.infer.DogsBreedClassifier.load_from_checkpoint")
@patch("src.infer.load_image")
@patch("src.infer.infer")
@patch("src.infer.save_and_display_prediction_image")
@patch("random.sample")
@patch("pathlib.Path.glob")
@patch("hydra.core.hydra_config.HydraConfig.get")
def test_main(mock_hydra_get, mock_glob, mock_sample, mock_save, mock_infer, mock_load_image, mock_load_model, mock_get_checkpoint, mock_hydra_config):
    mock_hydra_get.return_value = mock_hydra_config
    
    # Mock get_latest_checkpoint to return a path
    mock_checkpoint_path = str(Path(mock_hydra_config.paths.output_dir) / "checkpoints" / "latest_checkpoint.ckpt")
    mock_get_checkpoint.return_value = mock_checkpoint_path

    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_img = MagicMock()
    mock_img_tensor = MagicMock()
    mock_load_image.return_value = (mock_img, mock_img_tensor)
    mock_infer.return_value = ("Beagle", 0.95)

    # Mock glob to return some image paths
    mock_image_paths = [Path(mock_hydra_config.input_path) / f"image{i}.jpg" for i in range(10)]
    mock_glob.return_value = mock_image_paths

    # Mock random.sample to return a subset of these paths
    mock_sample.return_value = mock_image_paths[:mock_hydra_config.num_images]

    with patch("builtins.print"), patch("pathlib.Path.mkdir"):
        main(mock_hydra_config)

    mock_get_checkpoint.assert_called_once_with(mock_hydra_config.paths.output_dir)
    mock_load_model.assert_called_once_with(mock_checkpoint_path, map_location=torch.device("cpu"))
    assert mock_load_image.call_count == mock_hydra_config.num_images
    assert mock_infer.call_count == mock_hydra_config.num_images
    assert mock_save.call_count == mock_hydra_config.num_images

    # Check if the output file paths are correct
    for i, call in enumerate(mock_save.call_args_list):
        _, _, _, output_path = call[0]
        expected_path = Path(mock_hydra_config.output_path) / f"image{i}_prediction.png"
        assert output_path == expected_path

if __name__ == "__main__":
    pytest.main([__file__])

    