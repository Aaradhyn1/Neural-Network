from __future__ import annotations
import torch
import optuna
import wandb
import pytorch_lightning as pl
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# --- DATA COMPONENT ---
class LitDataModule(pl.LightningDataModule):
    def __init__(self, samples: int = 2000, features: int = 4, batch_size: int = 32):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage: str | None = None):
        x = torch.randn(self.hparams.samples, self.hparams.features)
        weights = torch.linspace(1.5, -1.5, steps=self.hparams.features)
        logits = (x @ weights) + 0.5
        y = (logits > torch.quantile(logits, 0.5)).float().unsqueeze(1)
        
        dataset = TensorDataset(x, y)
        train_size = int(0.8 * len(dataset))
        self.train_ds, self.val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.hparams.batch_size)

# --- MODEL COMPONENT ---
class LitFeedForward(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, lr: float = 1e-3, dropout: float = 0.1):
        super().__init__()
        self.save_hyperparameters() # Automatically saves these for loading later
        
        layers = [nn.Linear(input_size, hidden_size), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_size, 1))
        
        self.net = nn.Sequential(*layers)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor: return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = ((logits > 0).float() == y).float().mean()
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self): return optim.AdamW(self.parameters(), lr=self.hparams.lr)





def objective(trial: optuna.Trial):
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden = trial.suggest_categorical("hidden_size", [32, 64, 128])
    
    # 2. Setup Loggers & Callbacks
    wandb_logger = WandbLogger(project="lightning-optuna-v2", group="hpo", reinit=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, filename="best-{epoch}")
    
    # 3. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_acc", mode="max", patience=3)],
        accelerator="auto",
        devices=1,
        enable_checkpointing=True
    )
    
    model = LitFeedForward(input_size=4, hidden_size=hidden, lr=lr)
    datamodule = LitDataModule(features=4)
    
    trainer.fit(model, datamodule=datamodule)
    
    # Store best path in trial for easy retrieval later
    trial.set_user_attr("best_model_path", checkpoint_callback.best_model_path)
    return checkpoint_callback.best_model_score.item()

# Execute Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)






# Load the best model from the best trial's saved path
best_path = study.best_trial.user_attrs["best_model_path"]
best_model = LitFeedForward.load_from_checkpoint(best_path)

print(f"Best Model Loaded from: {best_path}")
print(f"Best Accuracy: {study.best_value:.4f}")

