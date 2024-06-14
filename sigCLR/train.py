from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
from torchsig.datasets.sig53 import Sig53
from torch.utils.data import DataLoader
import torchsig.transforms as ST
import torch
import os
import click
from sigdata import SigDataCLR
from sigclr import SigCLR


torch.set_float32_matmul_precision('medium')
num_workers = os.cpu_count()//4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True

print(f"Using device: {device}")
print(f"Number of workers: {num_workers}")

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

CHECKPOINT_PATH = "./saved_models/"

@click.command()
@click.option('--batch_size', default=32, help='Batch size used during training and validation.')
@click.option('--hidden_dim', default=53, help='Dimension of the hidden layer.')
@click.option('--epochs', default=100, help='Number of epochs during training.')
@click.option('--lr', default=0.001, help='Learning rate for the optimizer.')
@click.option('--weight_decay', default=1e-4, help='Weight_decay for the optimizer.')
@click.option('--temperature', default=0.07, help='Temperature rate used for ntXent loss computation.')
@click.option('--val_every', default=10, help='Run validation every val_every epochs.')
def train_sigclr(hidden_dim=53, lr=0.0001, temperature=0.07, weight_decay=1e-4, batch_size=64, epochs=500,device='cuda',val_every=10):
    accel="gpu" if str(device) == "cuda" else "cpu"
    checkpoint_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_top_k=3)
    trainer = Trainer(
        default_root_dir=CHECKPOINT_PATH,
        devices=-1,
        accelerator=accel,
        max_epochs=epochs,
        check_val_every_n_epoch=val_every,
        callbacks=[
            checkpoint_callback,
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
        model = SigCLR(hidden_dim=hidden_dim, lr=lr, temperature=temperature, weight_decay=weight_decay, batch_size=batch_size, max_epochs=epochs,device=device)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SigCLR.load_from_checkpoint(checkpoint_callback.best_model_path)

    return model

if __name__ == "__main__":
    sigclr_model = train_sigclr()
