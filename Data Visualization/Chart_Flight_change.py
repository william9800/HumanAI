import matplotlib.pyplot as plt

# 数据
x_labels = ['0.1','0.3','0.5','0.7','0.9','1.0','1.1','1.3','1.5','1.7','1.9']

data = {
    'EfficientNet-B7 with multi date': [100.00,100.00,100.00,99.80,100.00,98.50,94.81,85.11],
    'EfficientNet-B7 with data processing': [97.80,71.23,7.49,0.10,0.10,0.10,0.10,0.10],
    'EfficientNet-B7': [57.74,3.30,0.30,0.10,0.10,0.10,0.10,0.10],
    'VGG-16': [4.80,0.10,0.10,0.10,0.10,0.10,0.10,0.10],
    'ResNet-50': [36.26,0.2,0.1,0.1,0.1,0.1,0.1,0.1],
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
plt.title('Comparison of detection performance of each model under different Brightness Factor (using 1000 Stylegan3+1FFHQ)', fontsize=16)
plt.xlabel('Brightness Factor', fontsize=14)
plt.ylabel('Detection accuracy (%)', fontsize=14)
plt.ylim(0, 110)  # Y轴范围稍微高于100

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# 保存 & 显示
plt.savefig('Brightness_accuracy_curve.png', dpi=300)
plt.show()
