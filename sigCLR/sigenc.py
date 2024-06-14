from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
import torch.nn as nn
import torch

class SigEnc(nn.Module):
    def __init__(self, hidden_dim,pretrained=True,path="/project/def-msteve/torchsig-pretrained-models/sig53/efficientnet_b4_online.pt",device='cuda'):
        super().__init__()
        self.convnet = efficientnet_b4(pretrained=pretrained, path=path)

        # Add the projection head
        fc_input_size=self.convnet.classifier.in_features
        fc_output_size=self.convnet.classifier.out_features
        if hidden_dim != fc_output_size: # add extra layer to match the number of classes
            self.convnet.classifier = nn.Sequential(
                nn.Linear(fc_input_size, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim,fc_output_size,bias=False),
                #self.convnet.classifier,
            )
        else:
            self.convnet.classifier = nn.Sequential(
                nn.Linear(fc_input_size, hidden_dim),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dim,fc_output_size,bias=False),
                #self.convnet.classifier,
            )

        self.convnet = self.convnet.to(device)
    def forward(self, x):
        return self.convnet(x)

    def predict(self, x):
        with torch.no_grad():
            out = self.forward(x)
        return out

