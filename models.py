import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.models as models

# -- define custom cnn model -- #
class Custom(nn.Module):
    def __init__(self, num_classes):
        super(Custom, self).__init__()
        # block 1 - learns low level features (edges, vessel boundaries)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),       # 3 input channels (RGB) -> 32 feature maps
            nn.BatchNorm2d(32),                   # stabilises training and accelerates convergence
            nn.ReLU(),                            # introduces non-linearity
            nn.MaxPool2d(2, 2)                    # reduces spatial dimensions
        )
        # block 2 - learns intermediate patters (textures, lesions)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),      # increase feature depth to 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # block 3 - learns high level features (disease patterns)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(128, 256)            # fully connected layers map extracted
        self.fc2 = nn.Linear(256, 128)            # features to class predictions
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)            # reduce overfitting by randomly disabling neurons
        self.gap = nn.AdaptiveAvgPool2d((1,1))    # reduce each feature map to a single value

    # passing input data through the feature extractor
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)                            # global pooling to reduce spatial dimensions
        x = torch.flatten(x, 1)                    # flatten feature maps into 1D feature vector
        x = self.dropout(f.relu(self.fc1(x)))      # dropout regularisation
        x = self.dropout(f.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
def resnet18(num_classes, pretrained=True, dropout=0.0):
    # loading pretrained or randomly initialised resnet18 model
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    # replacing final classification layer to match dataset classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),            # feature compression, learning task-specific representation
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )

    return model

# layer freezing - controlling which parts of the pretrained model are trainable
def set_trainable_layers(model, mode="full"):
    # full fine-tuning - all layers are updated during training
    if mode == "full":
        # everything trainable
        for param in model.parameters():
            param.requires_grad = True
    # feature extraction frozen - only classifier head is trained
    elif mode == "freeze":
        # freeze all
        for param in model.parameters():
            param.requires_grad = False
        
        # unfreeze classifier head - allow training of final classification layer
        for param in model.fc.parameters():
            param.requires_grad = True
    # partial fine-tuning - early layers are frozen
    elif mode == "partial":
        # freeze all
        for param in model.parameters():
            param.requires_grad = False
        
        # unfreeze last block and head
        for param in model.layer4.parameters():
            param.requires_grad = True

        # unfreeze classifier head
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model
