from torch.utils.data import TensorDataset, DataLoader
from models.MLP import MyNeuralNet
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import wandb
import hydra
import os


@hydra.main(version_base="1.3", config_name="config.yaml", config_path="../")
def main(config):
    print(config)

    wandb_logger = WandbLogger(log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode="max")
    model = MyNeuralNet(config)

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=config.hyperparameters.epochs,
    )
    trainer.fit(model)

    return


if __name__ == "__main__":
    main()
