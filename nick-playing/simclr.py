from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from pytorch_lightning.callbacks import ModelCheckpoint
# from torchsig.utils.cm_plotter import plot_confusion_matrix
from pytorch_lightning import LightningModule, Trainer
from sklearn.metrics import classification_report
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import torchsig.transforms as ST
import numpy as np
import torchsig
import torch
import os
from torchvision import transforms
import random
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

train_dataloader = DataLoader(
    dataset=sig53_clean_train,
    batch_size=8,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
)

val_dataloader = DataLoader(
    dataset=sig53_clean_val,
    batch_size=8,
    num_workers=num_workers,
    shuffle=False,
    drop_last=True,
)


for iq_data, labels in iter(train_dataloader):
    fig = make_subplots(rows=4, cols=4, shared_xaxes=True,
                        subplot_titles=[
                            Sig53.convert_idx_to_name(labels[0][0].item()),
                            Sig53.convert_idx_to_name(labels[0][1].item()),
                            Sig53.convert_idx_to_name(labels[0][2].item()),
                            Sig53.convert_idx_to_name(labels[0][3].item()),
                            "",
                            "",
                            "",
                            "",
                            Sig53.convert_idx_to_name(labels[0][4].item()),
                            Sig53.convert_idx_to_name(labels[0][5].item()),
                            Sig53.convert_idx_to_name(labels[0][6].item()),
                            Sig53.convert_idx_to_name(labels[0][7].item()),
                            "",
                            "",
                            "",
                            ""
                        ],
                        specs=[
                            [{"r":0.015},{"l":0.015, "r":0.015},{"l":0.015, "r":0.015},{"l":0.015}],
                            [{"r":0.015,"b":0.05},{"l":0.015, "r":0.015, "b":0.05},{"l":0.015, "r":0.015, "b":0.05},{"l":0.015, "b": 0.05}],
                            [{"r":0.015, "t":0.025},{"l":0.015, "r":0.015, "t":0.025},{"l":0.015, "r":0.015, "t":0.025},{"l":0.015, "t":0.025}],
                            [{"r":0.015},{"l":0.015, "r":0.015},{"l":0.015, "r":0.015},{"l":0.015}],
                        ],
                        vertical_spacing=0.0,
                        horizontal_spacing=0.0
                        )
    
    row = 1
    col = 1
    for i in range(iq_data.shape[0]):
        fig.add_trace(
            go.Scatter(
                y=np.real(iq_data[i, 0].numpy()),
                mode="lines",
                line_color='blue',

            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                y=np.imag(iq_data[i, 0].numpy()),
                mode="lines",
                line_color='orange',

            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                y=np.real(iq_data[i, 1].numpy()),
                mode="lines",
                line_color='blue',
            ),
            row=row+1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                y=np.imag(iq_data[i, 1].numpy()),
                mode="lines",
                line_color='orange',
            ),
            row=row+1,
            col=col,
        )
        fig.update_layout()
        col += 1
        if col > 4:
            row += 2
            col = 1
    

    break
fig.update_layout(xaxis5_showticklabels=True, xaxis6_showticklabels=True, xaxis7_showticklabels=True, xaxis8_showticklabels=True, showlegend=False)
fig.show()