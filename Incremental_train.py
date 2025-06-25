#完成增量训练任务
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from dataset.dataset import HumanAI
from tqdm import tqdm
from torch.utils.data import ConcatDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强
transform = transforms.Compose([
    transforms.CenterCrop(600),  # EfficientNet-B7推荐尺寸
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 清理损坏图
def collate_fn(batch):
    batch = [x for x in batch if x[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# 训练函数
def train(model, loader, optimizer, criterion, epoch, scaler, use_amp=True):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    bar = tqdm(loader, desc=f"🚀 Fine-tune Epoch {epoch}")

    for data, target in bar:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        bar.set_postfix(loss=loss.item(), acc=correct / total * 100)

    avg_loss = total_loss / total
    acc = correct / total * 100
    print(f"✅ Epoch {epoch} 完成 | 平均损失: {avg_loss:.4f}, 准确率: {acc:.2f}%")

# 验证函数
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = correct / total * 100
    print(f"🎯 验证集准确率: {acc:.2f}%，平均损失: {total_loss / total:.4f}")
    return acc

# 主函数：调用入口
def finetune(base_model_path, old_data_dir, new_data_dir, val_data_dir=None,
             freeze_backbone=True, epochs=5, batch_size=32, save_path='model_finetuned.pth'):

    print(f"\n📥 加载模型: {base_model_path}")
    model = EfficientNet.from_name('efficientnet-b7')
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(torch.load(base_model_path, map_location=DEVICE))
    model.to(DEVICE)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model._fc.parameters():
            param.requires_grad = True
        print("🧊 冻结主干，仅微调全连接层")

    # 🔁 加载旧数据 + 新数据
    dataset_old = HumanAI(old_data_dir, transforms=transform, train=True)
    dataset_new = HumanAI(new_data_dir, transforms=transform, train=True)

    # ✅ 合并为一个训练集
    combined_dataset = ConcatDataset([dataset_old, dataset_new])

    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    if val_data_dir:
        val_dataset = HumanAI(val_data_dir, transforms=transform, train=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    else:
        val_loader = None


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = torch.amp.GradScaler()

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, scaler)
        if val_loader:
            validate(model, val_loader, criterion)
        scheduler.step()

    torch.save(model.state_dict(), save_path)
    print(f"\n✅ 增量训练完成，模型已保存为: {save_path}")

# 直接运行命令行
if __name__ == '__main__':
    # 示例调用（可修改）
    finetune(
        base_model_path="model_stylegan2.pth",
        old_data_dir="data/train",
        new_data_dir="data/train_stablediffusion",
        val_data_dir="data/test",
        freeze_backbone=False,
        epochs=5,
        batch_size=32,
        save_path="model_stylegan2+sd_mixed.pth"
    )