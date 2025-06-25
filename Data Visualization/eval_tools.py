from dataset.dataset import HumanAI
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EfficientNet.from_name('efficientnet-b7')
model._fc = torch.nn.Linear(model._fc.in_features, 2)
model.load_state_dict(torch.load('checkpoints/model_stylegan2_sd.pth'))
model.to(DEVICE)
model.eval()

sources = {
    "StyleGAN2": "data/stylegan2/test",
    "StableDiffusion": "data/stablediff/test",
    "JPEG": "data/attacks/jpeg",
    "Noise": "data/attacks/noise",
    "Blur": "data/attacks/blur"
}

transform = transforms.Compose([
    transforms.CenterCrop(600),  # EfficientNet-B7推荐尺寸
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

results = {}

for name, path in sources.items():
    ds = HumanAI(path, transforms=transform, test=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    correct, total = 0, 0
    for x, y in tqdm(loader, desc=name):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = 100. * correct / total
    results[name] = acc
    print(f"✅ [{name}] 准确率: {acc:.2f}%")

# 可选: 保存CSV
import pandas as pd
pd.DataFrame(results.items(), columns=["Source", "Accuracy (%)"]).to_csv("robustness_eval.csv", index=False)
