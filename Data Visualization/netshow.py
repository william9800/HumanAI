import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

model_ft = EfficientNet.from_pretrained('efficientnet-b7')
model_ft.eval()
print(model_ft)