import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import models, transforms
from dataset.dataset import HumanAI
from tqdm import tqdm

# ========= 全局设置 ========= #
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modellr = 1e-4
BATCH_SIZE = 8
EPOCHS = 10
USE_AMP = True
CHECKPOINT_DIR = "checkpoints"

# ========= 数据预处理 ========= #
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def collate_fn(batch):
    batch = [x for x in batch if x[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# ========= 模型主干列表 ========= #
backbones = {
    "efficientnet_b7": models.efficientnet_b7,
    "vgg16": models.vgg16,
    "resnet50": models.resnet50
}

# ========= 训练过程 ========= #
def train(model, loader, optimizer, scaler, epoch, criterion, name):
    model.train()
    total_loss, correct, total = 0, 0, 0
    train_bar = tqdm(loader, desc=f"[{name}] Epoch {epoch}")

    for x, y in train_bar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, pred = torch.max(out, 1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        total_loss += loss.item() * x.size(0)
        train_bar.set_postfix(loss=total_loss/total, acc=correct/total*100)

    return total_loss / total

def val(model, loader, criterion, name):
    model.eval()
    loss_total, correct, total = 0, 0, 0
    with torch.no_grad():
        test_bar = tqdm(loader, desc=f"[{name}] Val")
        for x, y in test_bar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(x)
                loss = criterion(out, y)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            loss_total += loss.item() * x.size(0)
            test_bar.set_postfix(loss=loss_total/total, acc=correct/total*100)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 加载数据集
    train_set = HumanAI("E:/HumanAI/data/train", transforms=transform, train=True)
    val_set = HumanAI("E:/HumanAI/data/test", transforms=transform, train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_set.classes)
    print(f"📌 共 {num_classes} 类别: {train_set.classes}")

    for name, backbone_fn in backbones.items():
        print(f"🚀 正在初始化: {name}")
        model = backbone_fn(pretrained=True)
        if "efficientnet" in name:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif "vgg" in name:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif "resnet" in name:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=modellr)
        scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

        for epoch in range(1, EPOCHS + 1):
            avg_loss = train(model, train_loader, optimizer, scaler, epoch, criterion, name)
            val(model, val_loader, criterion, name)

        # 保存模型
        model_path = os.path.join(CHECKPOINT_DIR, f"model_{name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ 已保存权重至: {model_path}")

if __name__ == "__main__":
    main()
