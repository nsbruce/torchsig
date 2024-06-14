from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from pytorch_lightning.callbacks import ModelCheckpoint
from torchsig.utils.cm_plotter import plot_confusion_matrix
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import classification_report
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch import Tensor, optim
from tqdm import tqdm
import torch.nn.functional as F
import torchsig.transforms as ST
import numpy as np
import torchsig
import torch
import os
import random
import torch.nn as nn

root = "/project/def-msteve/torchsig/sig53/"
train = True
impaired = False
class_list = list(Sig53._idx_to_name_dict.values())


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        transforms = random.sample(self.base_transforms, self.n_views)
        # return ST.Compose(transforms)(x)
        # return [self.base_transforms(x) for _ in range(self.n_views)]
        return [t(x) for t in transforms]

contrast_transforms = [
    ST.TimeVaryingNoise(),
    ST.RandomPhaseShift(),
    ST.TimeReversal(),
    ST.RandomTimeShift(),
    # ST.TimeCrop(),
    ST.GainDrift(),
    ST.LocalOscillatorDrift(),
    ST.Clip(),
    ST.SpectralInversion()
]
target_transform = ST.DescToClassIndex(class_list=class_list)


# Instantiate the Sig53 Clean Training Dataset
sig53_clean_train = Sig53(
    root=root, 
    train=train, 
    impaired=impaired,
    transform=ContrastiveTransformations(contrast_transforms, n_views=2),
    target_transform=target_transform,
    use_signal_data=True,
)

# Instantiate the Sig53 Clean Validation Dataset
train = False
sig53_clean_val = Sig53(
    root=root, 
    train=train, 
    impaired=impaired,
    transform=ContrastiveTransformations(contrast_transforms, n_views=2),
    target_transform=target_transform,
    use_signal_data=True,
)

# Set training parameters
num_workers = os.cpu_count()
# this might need to be bigger - SimCLR seems to benifit from large batch sizes
batch_size=256
#? I'm not sure how to pick this. In a SimCLR tutorial online for images it was 128  
hidden_dim=128 
max_epochs=500

# Create dataloaders
train_dataloader = DataLoader(
    dataset=sig53_clean_train,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset=sig53_clean_val,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=True,
)

# In the SimCLR paper, the base network f(.) is resnet50. I have a pretrained efficientnet_b4 model which is also convolutional and works on IQ data so I select it
model = efficientnet_b4(
    pretrained=True,
    path="/project/def-msteve/torchsig-pretrained-models/sig53/efficientnet_b4_online.pt",
)

# MLP for g(.) in the paper is Linear->ReLU->Linear. I hope this is right
model.classifier = nn.Sequential(
    model.classifier,
    nn.ReLU(inplace=True),
    nn.Linear(53 * hidden_dim, hidden_dim),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


class SimCLRNetwork(LightningModule):
    def __init__(self, model, data_loader, val_data_loader, lr=5e-4, temperature=0.07, weight_decay=1e-6, max_epochs=500):
        super(SimCLRNetwork, self).__init__()
        # self.save_hyperparameters(ignore=['model'])

        self.mdl = model
        self.data_loader = data_loader
        self.val_data_loader = val_data_loader

        # save hparams
        self.hparams.lr = lr
        self.hparams.temperature = temperature
        self.hparams.weight_decay = weight_decay
        self.hparams.max_epochs = max_epochs


    def forward(self, x):
        return self.mdl(x)

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return self.data_loader

    def info_nce_loss(self, batch: list[Tensor, list[Tensor]], mode="train"):
        """Normalized cross-entropy is the loss function for SimCLR"""
        sigs, _ = batch

        # Encode all images
        feats = self.mdl(sigs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_nb):
        loss = self.info_nce_loss(batch, mode='train')
        return {"loss": loss}

    def val_dataloader(self):
        return self.val_data_loader

    def validation_step(self, batch, batch_nb):
        val_loss = self.info_nce_loss(batch, mode="val")
        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}
    
network_model = SimCLRNetwork(model, train_dataloader, val_dataloader)


# Setup checkpoint callbacks
checkpoint_filename = "{}/checkpoints/checkpoint".format(os.getcwd())
checkpoint_callback = ModelCheckpoint(
    filename=checkpoint_filename,
    save_top_k=True,
    monitor="val_loss",
    mode="min",
)
torch.set_float32_matmul_precision('medium')
trainer = Trainer(
    max_epochs=max_epochs, callbacks=checkpoint_callback, accelerator="gpu", devices=1
)
trainer.fit(network_model)

# Load best checkpoint
checkpoint = torch.load(checkpoint_filename+".ckpt", map_location=lambda storage, loc: storage)
network_model.load_state_dict(checkpoint["state_dict"], strict=False)
network_model = network_model.eval()
network_model = network_model.cuda() if torch.cuda.is_available() else network_model
