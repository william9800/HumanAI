import matplotlib.pyplot as plt

# 数据
x_labels = ['95%', '80%', '60%', '40%', '20%', '10%', '5%']

data = {
    'EfficientNet-B7 with multi date': [100, 100, 99.6, 99.0, 88.81, 36.56, 3.90],
    'EfficientNet-B7 with data processing': [99.80, 98.70, 98.30, 98.40, 95.30, 83.72, 49.25],
    'EfficientNet-B7': [97.00, 78.32, 48.95, 31.47, 14.19, 0.70, 0.10],
    'VGG-16': [99.20, 98.01, 73.53, 53.65, 12.89, 0.10, 0.10],
    'ResNet-50': [95.10, 78.72, 42.66, 8.39, 0.10, 0.10, 0.10]
}

# 颜色列表（可调整）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

plt.figure(figsize=(12, 7))

for i, (model, values) in enumerate(data.items()):
    plt.plot(x_labels, values, marker='o', label=model, color=colors[i])
    # 每个点上标注数值
    for x, y in zip(x_labels, values):
        plt.text(x, y + 1, f'{y:.2f}%', ha='center', va='bottom', fontsize=9)

# 标题与标签
plt.title('Comparison of detection performance of each model under different compression rates (using 1000 Stylegan+1FFHQ)', fontsize=16)
plt.xlabel('Compression Rate', fontsize=14)
plt.ylabel('Detection accuracy (%)', fontsize=14)
plt.ylim(0, 110)  # Y轴范围稍微高于100

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# 保存 & 显示
plt.savefig('compression_accuracy_curve.png', dpi=300)
plt.show()
