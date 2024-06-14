from pytorch_lightning import LightningModule
from torch import optim
import torch.nn as nn
import torch
from sigenc import SigEnc


class SigCLR(LightningModule):
    def __init__(self, hidden_dim=53, lr=0.0001, temperature=0.07, weight_decay=1e-4, batch_size=64, max_epochs=500,device='cuda'):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        self.encoder = SigEnc(hidden_dim)
        self.temperature = temperature
        self.batch_size=batch_size
        self.hparams.device=device
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.device=device
        
        # Produce Mask to Mask the self when computing the loss
        self.allN = 2 * batch_size
        self.mask = torch.ones((self.allN, self.allN), dtype=bool,device=self.device)
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
        (xi,xj), _ = batch
        if xi.shape[0]!=self.batch_size: # Recompute the mask
            self.batch_size=xi.shape[0]
            self.allN=2*self.batch_size
            self.mask = torch.ones((self.allN, self.allN), dtype=bool,device=self.device)
            self.mask = self.mask.fill_diagonal_(0)
            for i in range(self.batch_size):
                self.mask[i, self.batch_size + i] = 0
                self.mask[self.batch_size + i, i] = 0
            
        zi,zj = self.forward(xi,xj)
        z = torch.cat((zi, zj), dim=0)
        sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(self.allN, 1)
        negative_samples = sim[self.mask].reshape(self.allN, -1)

        labels = torch.zeros(self.allN).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= self.allN

        # Logging loss
        self.log(mode + "_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.ntXent_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.ntXent_loss(batch, mode="val")