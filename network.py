import torch
import torch.nn as nn
from torchvision import models


class MVCNN(nn.Module):
    """
    Multi-View Convolutional Neural Network (MVCNN)
    Initializes a model with the architecture of a MVCNN with a ResNet34 base.
    """
    def __init__(self, num_classes=6, pretrained=True):
        super(MVCNN, self).__init__()
        resnet = models.resnet34(pretrained = pretrained)
        fc_in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, inputs): # inputs.shape = samples x views x height x width x channels
        inputs = inputs.transpose(0, 1)
        view_features = [] 
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)   
            
        pooled_views, _ = torch.max(torch.stack(view_features), 0)
        outputs = self.classifier(pooled_views)
        return outputs