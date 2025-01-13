import wandb
import torch
from mlops_myproject.model import MyAwesomeModel

run = wandb.init(project="corrupt_mnist", entity="s183587-danmarks-tekniske-universitet-dtu")
artifact = run.use_artifact('s183587-danmarks-tekniske-universitet-dtu/corrupt_mnist/corrupt_mnist_model:v0', type='model')
artifact_dir = artifact.download()
model = MyAwesomeModel()
model.load_state_dict(torch.load(f"{artifact_dir}/model.ckpt"))