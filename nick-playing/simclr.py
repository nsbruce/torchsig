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

torch.set_float32_matmul_precision('medium')
num_workers = os.cpu_count()
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
# Specify Sig53 Options
root = "/data/torchsig-datasets/sig53/"
train = True
impaired = False
class_list = list(Sig53._idx_to_name_dict.values())

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

# # Retrieve a sample and print out information to verify
# idx = np.random.randint(len(sig53_clean_train))
# data, label = sig53_clean_train[idx]
# print("Dataset length: {}".format(len(sig53_clean_train)))
# print("Data shape: {}".format(data.shape))
# print("Label Index: {}".format(label))
# print("Label Class: {}".format([Sig53.convert_idx_to_name(l) for l in label]))

# train_dataloader = DataLoader(
#     dataset=sig53_clean_train,
#     batch_size=8,
#     num_workers=num_workers,
#     shuffle=True,
#     drop_last=True,
# )

# val_dataloader = DataLoader(
#     dataset=sig53_clean_val,
#     batch_size=8,
#     num_workers=num_workers,
#     shuffle=False,
#     drop_last=True,
# )


# for iq_data, labels in iter(train_dataloader):
#     fig = make_subplots(rows=4, cols=4, shared_xaxes=True,
#                         subplot_titles=[
#                             Sig53.convert_idx_to_name(labels[0][0].item()),
#                             Sig53.convert_idx_to_name(labels[0][1].item()),
#                             Sig53.convert_idx_to_name(labels[0][2].item()),
#                             Sig53.convert_idx_to_name(labels[0][3].item()),
#                             "",
#                             "",
#                             "",
#                             "",
#                             Sig53.convert_idx_to_name(labels[0][4].item()),
#                             Sig53.convert_idx_to_name(labels[0][5].item()),
#                             Sig53.convert_idx_to_name(labels[0][6].item()),
#                             Sig53.convert_idx_to_name(labels[0][7].item()),
#                             "",
#                             "",
#                             "",
#                             ""
#                         ],
#                         specs=[
#                             [{"r":0.015},{"l":0.015, "r":0.015},{"l":0.015, "r":0.015},{"l":0.015}],
#                             [{"r":0.015,"b":0.05},{"l":0.015, "r":0.015, "b":0.05},{"l":0.015, "r":0.015, "b":0.05},{"l":0.015, "b": 0.05}],
#                             [{"r":0.015, "t":0.025},{"l":0.015, "r":0.015, "t":0.025},{"l":0.015, "r":0.015, "t":0.025},{"l":0.015, "t":0.025}],
#                             [{"r":0.015},{"l":0.015, "r":0.015},{"l":0.015, "r":0.015},{"l":0.015}],
#                         ],
#                         vertical_spacing=0.0,
#                         horizontal_spacing=0.0
#                         )
    
#     row = 1
#     col = 1
#     for i in range(iq_data.shape[0]):
#         fig.add_trace(
#             go.Scatter(
#                 y=np.real(iq_data[i, 0].numpy()),
#                 mode="lines",
#                 line_color='blue',

#             ),
#             row=row,
#             col=col,
#         )
#         fig.add_trace(
#             go.Scatter(
#                 y=np.imag(iq_data[i, 0].numpy()),
#                 mode="lines",
#                 line_color='orange',

#             ),
#             row=row,
#             col=col,
#         )
#         fig.add_trace(
#             go.Scatter(
#                 y=np.real(iq_data[i, 1].numpy()),
#                 mode="lines",
#                 line_color='blue',
#             ),
#             row=row+1,
#             col=col,
#         )
#         fig.add_trace(
#             go.Scatter(
#                 y=np.imag(iq_data[i, 1].numpy()),
#                 mode="lines",
#                 line_color='orange',
#             ),
#             row=row+1,
#             col=col,
#         )
#         fig.update_layout()
#         col += 1
#         if col > 4:
#             row += 2
#             col = 1
    

#     break
# fig.update_layout(xaxis5_showticklabels=True, xaxis6_showticklabels=True, xaxis7_showticklabels=True, xaxis8_showticklabels=True, showlegend=False)
# fig.show()

class SimCLR(LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        # self.convnet = resnet18(
        #     pretrained=False, num_classes=4 * hidden_dim
        # )  # num_classes is the output size of the last linear layer
        self.convnet = efficientnet_b4(pretrained=True, path="/data/torchsig-pretrained-models/sig53/efficientnet_b4_online.pt")
        # The MLP for g(.) consists of Linear->ReLU->Linear
        # self.convnet.fc = nn.Sequential(
        #     self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4 * hidden_dim, hidden_dim),
        # )
        # I think this is equivalent
        self.convnet.classifier = nn.Sequential(
            self.convnet.classifier,
            nn.ReLU(inplace=True),
            nn.Linear(53 * hidden_dim, hidden_dim),
        )
        self.convnet = self.convnet.to(device)

    def forward(self, x):
        return self.convnet(x)

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

    def info_nce_loss(self, batch: list[Tensor, list[Tensor]], mode="train"):
        sigs, _ = batch

        # Encode all images
        feats = self.convnet(sigs)
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

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="val")


CHECKPOINT_PATH = "/workspaces/torchsig/nick-playing/saved_models/"

def train_simclr(batch_size, max_epochs=500, **kwargs):
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
        model = SimCLR.load_from_checkpoint(pretrained_filename)
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
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model

simclr_model = train_simclr(batch_size=256, hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=500)