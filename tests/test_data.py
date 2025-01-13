import os
import pytest
import torch
from mlops_myproject.data import corrupt_mnist

train_images_path = "/home/lchristiansen/mlops_template/data/processed/train_images.pt"
train_target_path = "/home/lchristiansen/mlops_template/data/processed/train_target.pt"
test_images_path = "/home/lchristiansen/mlops_template/data/processed/test_images.pt"
test_target_path = "/home/lchristiansen/mlops_template/data/processed/test_target.pt"

@pytest.mark.skipif(
    not (os.path.exists(train_images_path) and os.path.exists(train_target_path) and os.path.exists(test_images_path) and os.path.exists(test_target_path)),
    reason="Data files not found"
)
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Train dataset did not have the correct number of samples"
    assert len(test) == 5000, "Test dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Input image shape is incorrect, expected (1, 28, 28)"
            assert y in range(10), "Target value is out of expected range (0-9)"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "Train dataset does not contain all classes (0-9)"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), "Test dataset does not contain all classes (0-9)"