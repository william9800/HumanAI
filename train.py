import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
from dataset.dataset import HumanAI
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

# 设备选择
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
modellr = 1e-4
BATCH_SIZE = 8
EPOCHS = 10
CHECKPOINT_PATH = "checkpoint_4.pth"
USE_AMP = True
PATIENCE = 5  # EarlyStopping耐心
REDUCE_LR_PATIENCE = 2  # 学习率调整耐心
LR_FACTOR = 0.5  # 每次降低到原来的多少倍

# 数据预处理
class AdditiveGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

transform_train = transforms.Compose([
    transforms.Resize((620, 620)),
    transforms.RandomResizedCrop(600, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    AdditiveGaussianNoise(mean=0., std=0.02),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def save_checkpoint(model, optimizer, epoch, loss, save_path=CHECKPOINT_PATH):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, save_path)
    print(f"💾 Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        print(f"🔄 恢复训练: 从 epoch {start_epoch} 继续，损失: {loss:.6f}")
        return start_epoch
    return 1

def train(model, device, train_loader, optimizer, epoch, scaler, criterion):
    model.train()
    sum_loss, correct, total = 0.0, 0, 0
    train_bar = tqdm(train_loader, desc=f'🚀 训练 Epoch [{epoch}/{EPOCHS}]')

    for batch_idx, (data, target) in enumerate(train_bar):
        data, target = data.to(device), target.to(device)
        target = target.float().unsqueeze(1)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        sum_loss += loss.item() * data.size(0)
        preds = torch.sigmoid(output).round()
        correct += (preds.cpu() == target.cpu()).sum().item()
        total += target.size(0)

        train_bar.set_postfix(loss=loss.item(), acc=correct / total * 100)

    avg_loss = sum_loss / total
    acc = correct / total * 100
    return avg_loss, acc

def validate(model, device, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='🎯 验证中')
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            target = target.float().unsqueeze(1)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                output = model(data)
                loss = criterion(output, target)

            test_loss += loss.item() * data.size(0)
            preds = torch.sigmoid(output).round()
            correct += (preds.cpu() == target.cpu()).sum().item()
            total += target.size(0)

            test_bar.set_postfix(loss=loss.item(), acc=correct / total * 100)

    avg_loss = test_loss / total
    acc = correct / total * 100
    return avg_loss, acc

def collate_fn(batch):
    batch = [x for x in batch if x[0] is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

if __name__ == '__main__':
    dataset_train = HumanAI('data/train', transforms=transform_train, train=True)
    dataset_test = HumanAI('data/test', transforms=transform_test, train=False)

    num_classes = len(dataset_train.classes)
    print(f"📌 发现 {num_classes} 个类别: {dataset_train.classes}")

    num_workers = 4 if os.name != 'nt' else 0
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model_ft = EfficientNet.from_pretrained('efficientnet-b7')
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, 1)  # 二分类
    model_ft.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=modellr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_FACTOR, patience=REDUCE_LR_PATIENCE, verbose=True)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    start_epoch = load_checkpoint(model_ft, optimizer)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch, scaler, criterion)
        val_loss, val_acc = validate(model_ft, DEVICE, test_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        # 🔍 当前学习率打印
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📉 当前学习率: {current_lr:.6f}")

        save_checkpoint(model_ft, optimizer, epoch, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model_ft.state_dict(), 'model_best.pth')
            print(f"🌟 最佳模型已保存 (val loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"⏹️ Early stopping triggered at epoch {epoch}")
                break

    torch.save(model_ft.state_dict(), 'model_final.pth')
    print("✅ 训练完成，最终模型已保存: model_final.pth")

    # 📈 绘制曲线
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()
