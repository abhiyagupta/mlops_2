import pytest
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import rootutils
import shutil
from unittest.mock import patch, MagicMock, call
import lightning as L

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from src.eval import evaluate
from src.eval import (
    evaluate, 
    instantiate_callbacks, 
    instantiate_loggers, 
    get_latest_checkpoint,
    main
)

@pytest.fixture(scope="function")
def config():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="eval")
        return cfg

@patch('src.eval.get_latest_checkpoint')
def test_evaluate(mock_get_latest_checkpoint, config: DictConfig, tmp_path: Path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / "logs")

    # Mock the components
    mock_datamodule = MagicMock()
    mock_model = MagicMock()
    mock_trainer = MagicMock()

    # Mock hydra.utils.instantiate to return our mocks
    def mock_instantiate(cfg):
        if "datamodule" in str(cfg):
            return mock_datamodule
        elif "model" in str(cfg):
            return mock_model
        elif "trainer" in str(cfg):
            return mock_trainer
        return MagicMock()

    # Set up the mock for test_dataloader
    mock_test_loader = MagicMock()
    mock_datamodule.test_dataloader.return_value = mock_test_loader

    # Mock the trainer.test method to return a list with a single dictionary
    mock_trainer.test.return_value = [{"test_metric": 0.95}]

    # Mock get_latest_checkpoint to return a checkpoint path
    mock_checkpoint_path = str(tmp_path / "checkpoints" / "checkpoint.ckpt")
    mock_get_latest_checkpoint.return_value = mock_checkpoint_path

    with patch("hydra.utils.instantiate", side_effect=mock_instantiate):
        # Run evaluation
        metrics = evaluate(config, mock_trainer, mock_model, mock_datamodule)

    # Assertions
    assert mock_trainer.test.call_count == 1, "trainer.test should be called once"
    
    # Check each argument of the trainer.test call
    args, kwargs = mock_trainer.test.call_args
    assert kwargs['model'] == mock_model, "Wrong model passed to trainer.test"
    assert kwargs['dataloaders'] == mock_test_loader, "Wrong dataloaders passed to trainer.test"
    assert kwargs['ckpt_path'] == mock_checkpoint_path, f"Wrong ckpt_path passed to trainer.test. Expected {mock_checkpoint_path}, got {kwargs['ckpt_path']}"

    # Check if metrics are returned correctly
    assert metrics == [{"test_metric": 0.95}], "Evaluate should return the metrics from trainer.test"

def test_instantiate_callbacks():
    callback_cfg = DictConfig({"callback1": {"_target_": "some.callback.Class"}})
    with patch("hydra.utils.instantiate") as mock_instantiate:
        callbacks = instantiate_callbacks(callback_cfg)
    assert len(callbacks) == 1
    mock_instantiate.assert_called_once()

def test_instantiate_loggers():
    logger_cfg = DictConfig({"logger1": {"_target_": "some.logger.Class"}})
    with patch("hydra.utils.instantiate") as mock_instantiate:
        loggers = instantiate_loggers(logger_cfg)
    assert len(loggers) == 1
    mock_instantiate.assert_called_once()

@pytest.mark.parametrize("checkpoint_exists", [True, False])
def test_get_latest_checkpoint(tmpdir, checkpoint_exists):
    base_dir = Path(tmpdir)
    ckpt_dir = base_dir / "checkpoints"
    # Clean up any existing checkpoints
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    if checkpoint_exists:
        ckpt_dir.mkdir(parents=True)
        checkpoint_file = ckpt_dir / "checkpoint.ckpt"
        checkpoint_file.touch()
        
        result = get_latest_checkpoint(base_dir)
        assert result == str(checkpoint_file)
    else:
        with pytest.raises(FileNotFoundError):
            get_latest_checkpoint(base_dir)

    # Clean up after the test
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

@patch('src.eval.get_latest_checkpoint')
def test_evaluate_without_checkpoint(mock_get_latest_checkpoint, config, tmp_path):
    config.paths.output_dir = str(tmp_path)
    mock_trainer = MagicMock()
    mock_model = MagicMock()
    mock_datamodule = MagicMock()
    
    # Simulate no checkpoint found
    mock_get_latest_checkpoint.side_effect = FileNotFoundError

    evaluate(config, mock_trainer, mock_model, mock_datamodule)
    
    mock_trainer.test.assert_called_once_with(model=mock_model, dataloaders=mock_datamodule.test_dataloader())

@patch("src.eval.instantiate_callbacks")
@patch("src.eval.instantiate_loggers")
@patch("hydra.utils.instantiate")
@patch("src.eval.evaluate")
def test_main(mock_evaluate, mock_instantiate, mock_instantiate_loggers, mock_instantiate_callbacks, config):
    mock_instantiate.side_effect = [MagicMock(spec=L.LightningDataModule), MagicMock(spec=L.LightningModule), MagicMock(spec=L.Trainer)]
    main(config)
    mock_evaluate.assert_called_once()
    assert mock_instantiate.call_count == 3
    mock_instantiate_loggers.assert_called_once()
    mock_instantiate_callbacks.assert_called_once()

# Add a simple test to ensure pytest is working
def test_pytest_is_working():
    assert True, "This test should always pass"