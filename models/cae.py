import timm
import numpy as np
import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, in_size=224, in_channels=3, encoder_name='resnet50d', embed_size=512, classes=24):
        super(CAE, self).__init__()

        self.encoder = timm.create_model(encoder_name, pretrained=False, num_classes=4096)
       
        self.embedding = nn.Sequential(
           nn.Linear(in_features=4096, out_features=embed_size, bias=False))
        
        self.classifier = nn.Sequential(   
           nn.Linear(in_features=embed_size, out_features=1024, bias=True),
           nn.ReLU(),       
           nn.Linear(in_features=1024, out_features=classes, bias=False))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def forward(self, input):
         encoder = self.encoder(input)        
         embedding = self.embedding(encoder)

         classifier = self.classifier(embedding)
         return embedding, classifier, self.logit_scale.exp()
    
