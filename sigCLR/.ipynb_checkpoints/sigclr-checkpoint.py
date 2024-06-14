from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from torchsig.utils.cm_plotter import plot_confusion_matrix
from pytorch_lightning import LightningModule, Trainer, seed_everything
from sklearn.metrics import classification_report
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch import Tensor, optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torchsig.transforms as ST
import numpy as np
import torchsig
import torch
import os
from torchvision import transforms
from torchvision.models import resnet18
import random
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sigenc import SigEnc
from sigdata import SigDataCLR

torch.set_float32_matmul_precision('medium')
num_workers = os.cpu_count()//4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True

print(f"Using device: {device}")
print(f"Number of workers: {num_workers}")



class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        transforms = random.sample(self.base_transforms, self.n_views)
        return ST.Compose([transforms[0],ST.ComplexTo2D()])(x), ST.Compose([transforms[1],ST.ComplexTo2D()])(x)

contrast_transforms = [
    ST.TimeVaryingNoise(),
    ST.RandomPhaseShift(),
    ST.TimeReversal(),
    ST.RandomTimeShift(),
    # ST.TimeCrop(),
    ST.GainDrift(),
    ST.LocalOscillatorDrift(),
    ST.Clip(),
    ST.SpectralInversion(),
]
# Specify Sig53 Options
root = "/project/def-msteve/torchsig/sig53/"
train = True
impaired = False
class_list = list(Sig53._idx_to_name_dict.values())

target_transform = ST.DescToClassIndex(class_list=class_list)

# Instantiate the Sig53 Clean Training Dataset
sig53_clean_train = SigDataCLR(Sig53(
    root=root, 
    train=train, 
    impaired=impaired,
    transform=None,
    target_transform=target_transform,
    use_signal_data=True,
), transforms=contrast_transforms)

# Instantiate the Sig53 Clean Validation Dataset
train = False
sig53_clean_val = SigDataCLR(Sig53(
    root=root, 
    train=train, 
    impaired=impaired,
    transform=None,
    target_transform=target_transform,
    use_signal_data=True,
),transforms=contrast_transforms)

class SigCLR(LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, batch_size=64, max_epochs=500,device='cuda'):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        self.encoder = SigEnc(hidden_dim)
        self.temperature = temperature
        self.batch_size=batch_size
        self.hparams.device=device
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
        # Produce Mask to Mask the self when computing the loss
        N = 2 * batch_size
        self.mask = torch.ones((N, N), dtype=bool,device=device)
        self.mask = self.mask.fill_diagonal_(0)
        for i in range(batch_size):
            self.mask[i, batch_size + i] = 0
            self.mask[batch_size + i, i] = 0
    def forward(self, xi, xj):
        zi, zj=self.encoder(xi),self.encoder(xj)
        return zi, zj

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def ntXent_loss(self, batch, mode="train"):
         #self.batch_size=batch.shape[0]
        assert self.batch_size == batch.shape[0], "data batch is not the same as the batch_size provide!"
        (xi,xj), _ = batch
        zi,zj = self.convnet(xi,xj)
        z = torch.cat((zi, zj), dim=0)
        sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        # Logging loss
        self.log(mode + "_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.ntXent_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.ntXent_loss(batch, mode="val")


CHECKPOINT_PATH = "./saved_models/"

def train_sigclr(batch_size, max_epochs=500, **kwargs):
    trainer = Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
        # gpus=1 if str(device) == "cuda:0" else 0,
        devices=-1,
        accelerator="gpu",
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
        # progress_bar_refresh_rate=1,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SigCLR.load_from_checkpoint(pretrained_filename)
    else:
        train_loader = DataLoader(
            sig53_clean_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            # pin_memory=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            sig53_clean_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            # pin_memory=True,
            num_workers=num_workers,
        )
        seed_everything(42)  # To be reproducable
        model = SigCLR(max_epochs=max_epochs, batch_size=batch_size,device=device, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SigCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model

sigclr_model = train_sigclr(batch_size=64, hidden_dim=53, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=500)
