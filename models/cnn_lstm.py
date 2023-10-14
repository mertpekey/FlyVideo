import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from transformers import AutoImageProcessor

from .base_model import BaseModel


class CnnLstm(BaseModel):
    def __init__(self, num_classes, num_frames, image_processor_ckpt):
        super(CnnLstm, self).__init__(num_classes, num_frames)

        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_ckpt)

        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 128))
        self.resnet.eval()

        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)
       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def _freeze_layers(self, freeze_type):
        pass

    def _load_pretrained_weights(self, pretrained_weights_path):
        pass