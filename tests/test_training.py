import torch
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from mlops_myproject.model import MyAwesomeModel
from mlops_myproject.train import train

def test_training():
    model = MyAwesomeModel()
    initial_params = [param.clone() for param in model.parameters()]
    x = torch.randn(10, 1, 28, 28)
    y = torch.randint(0, 10, (10,))
    train_dataloader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size=2)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", UserWarning)
        train(model, train_dataloader, epochs=1)
    
    updated_params = [param for param in model.parameters()]
    
    for initial, updated in zip(initial_params, updated_params):
        assert not torch.equal(initial, updated), "Model parameters did not update during training"