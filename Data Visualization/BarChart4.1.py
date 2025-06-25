import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# 准备数据
data = {
    "Data Combination": [
        '500stylegan3+500unsplashed',
        '21Dall3+21FFHQ',
        '40flux+40FFHQ',
        '100doubao+100FFHQ',
        '100SDXL+100FFHQ',
        '500stable+500FFHQ',
        '1000Lexica+1000FFHQ',
        '1000randomweb+1000FFHQ',
        '2000generated+2000FFHQ',
        '2000stylegan2+2000FFHQ',
        '2000stylegan+2000FFHQ',
        '2000stylegan+2000CelebA',
        '2000stylegan+2000FFHQ',
        '2000stylegan+2000Star'
    ],
    "EB7 Base": [99.9,	50,	50,	50,	50,	49.8,	49.95,	53.5,	50.52,	75.3,99.6 ,	96.05,	99.58,	99.65
                 ],
    "VGG-16":   [99.6,	50,	53.75,	50.5,	54,	52.8,	53.7,	68.35,	99.88,	67.15,99.4,	75.6,	99.62,	92.25
                 ],
    "ResNet-50":[99.8,	50,	50,	50,	50,	50,	50.05,	56.85,	81.23,	76.20,99.8, 	99.88,	99.92,	99.77
                 ]
}

df = pd.DataFrame(data)

# 设置可视化风格
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 24,
    'axes.titlesize': 20,
    'axes.labelsize': 24,
    'figure.titlesize': 24
})

# ========= 柱状图 =========
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(df))
width = 0.2
colors = ['#2ca02c', '#d62728', '#9467bd']

cols_to_plot = ['EB7 Base', 'VGG-16', 'ResNet-50']

for i, col in enumerate(cols_to_plot):
    bars = ax.bar(x + i * width, df[col], width, label=col, color=colors[i],
                  edgecolor='white', linewidth=0.5)

    # 添加交错内部标签
    for j, bar in enumerate(bars):
        height = bar.get_height()
        xpos = bar.get_x() + bar.get_width() / 2
        # 阈值: 柱子太矮时，放在顶部
        if height < 15:
            ax.text(xpos, height + 2, f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
        else:
            # 不同列交错显示
            if i == 0:
                y_pos = height * 0.75  # 偏上
            elif i == 1:
                y_pos = height * 0.5   # 正中
            else:
                y_pos = height * 0.3   # 偏下
            ax.text(xpos, y_pos, f'{height:.1f}%',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='black')

ax.set_xticks(x + width)
ax.set_xticklabels(df["Data Combination"], rotation=45, ha='right')
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(PercentFormatter())
ax.set_title('Comparison of the Accuracy of Different Models', pad=20)
ax.legend(loc='upper right', ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('model_comparison_inbar_labels.png', dpi=300)
plt.show()
