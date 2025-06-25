import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 准备数据
data = {
    'Model': ['SVM', 'ViT', 'EfficientNet-B7'],
    'Accuracy': [77, 90, 98.4],
    'AUC': [0.872, 0.968, 0.999],
    'AP': [0.895, 0.971, 0.999]
}
df = pd.DataFrame(data)

# 设置绘图风格
sns.set(style="whitegrid", font_scale=1.7)  # 调整全局字体大小
plt.figure(figsize=(12, 6))

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 准确率柱状图
sns.barplot(x='Model', y='Accuracy', data=df, ax=axes[0], palette="Blues_d")
axes[0].set_title('Accuracy Comparison (%)', fontsize=20)
axes[0].set_ylim(0, 100)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),  # 显示在柱子中间
                     ha='center', va='center',  # 水平和垂直居中
                     color='white',  # 文字颜色设为白色
                     fontsize=24, fontweight='bold')  # 调整字体大小和粗细

# AUC柱状图
sns.barplot(x='Model', y='AUC', data=df, ax=axes[1], palette="Greens_d")
axes[1].set_title('AUC Score Comparison', fontsize=20)
axes[1].set_ylim(0, 1.05)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.3f}",
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center',
                     color='white',
                     fontsize=24, fontweight='bold')

# AP柱状图
sns.barplot(x='Model', y='AP', data=df, ax=axes[2], palette="Reds_d")
axes[2].set_title('Average Precision Comparison', fontsize=20)
axes[2].set_ylim(0, 1.05)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.3f}",
                     (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                     ha='center', va='center',
                     color='white',
                     fontsize=24, fontweight='bold')

plt.tight_layout()
plt.savefig('compare-bar-tradional.png', dpi=300, bbox_inches='tight')
plt.show()