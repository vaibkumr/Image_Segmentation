from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def get_effnet_model(num_classes, model_name, freeze=True):
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)

    if freeze:
        # Transfer learning -- don't freeze last few ~5 layers. Train them.
        for i, param in list(model.named_parameters())[:-5]:
            param.requires_grad = False

    else:
        for param in model.parameters():
            param.requires_grad = True

    return model

class Effnet(nn.Module):
    def __init__(self, n_classes=4, model_name='efficientnet-b2', freeze=True):
        super(Effnet, self).__init__()
        self.model = get_effnet_model(n_classes, model_name, freeze)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sig(x)

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True



def get_model_inet(num_classes, pretrained=True, freeze=True):
    model = models.inception_v3(pretrained=pretrained)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    in_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_ftrs, num_classes)

    in_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(in_ftrs, num_classes)

    return model

class InceptionNet(nn.Module):
    def __init__(self, n_classes=4, pretrained=True, freeze=True):
        super(InceptionNet, self).__init__()
        self.inception = get_model_inet(n_classes, pretrained, freeze)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inception(x)
        if self.training:
            return self.sig(x[0]), self.sig(x[1])
        else:
            return self.sig(x)

    def unfreeze(self):
        for param in model.parameters():
            param.requires_grad = True
