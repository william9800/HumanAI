import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define a function to draw a block
def draw_block(ax, xy, width, height, label, color='lightblue'):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02",
        edgecolor='black',
        facecolor=color,
        linewidth=1
    )
    ax.add_patch(rect)
    rx, ry = xy
    cx = rx + width / 2.0
    cy = ry + height / 2.0
    ax.text(cx, cy, label, fontsize=8, ha='center', va='center')

# Create a more academic version of the EfficientNet-B0 architecture diagram
fig, ax = plt.subplots(figsize=(14, 4))

# Define block properties: (Label, Repeats, Kernel Size, Output Channels, Stride)
blocks = [
    ("Stem", "Conv3x3", 32, 2),
    ("MBConv1", "3x3", 16, 1),
    ("MBConv6 x2", "3x3", 24, 2),
    ("MBConv6 x2", "5x5", 40, 2),
    ("MBConv6 x3", "3x3", 80, 2),
    ("MBConv6 x3", "5x5", 112, 1),
    ("MBConv6 x4", "5x5", 192, 2),
    ("MBConv6", "3x3", 320, 1),
    ("Head", "Conv1x1", 1280, 1),
    ("Output", "Pooling + FC", "1000", "-")
]

x = 0
for label, ksize, out_ch, stride in blocks:
    width = 0.8
    detail = f"{ksize}, {out_ch}ch\nStride {stride}" if stride != "-" else f"{ksize}\n{out_ch} classes"
    draw_block(ax, (x, 0.5), width, 1.5, f"{label}\n{detail}", color='#cce5ff')
    x += width + 0.3

# Adjust plot aesthetics
ax.set_xlim(0, x)
ax.set_ylim(0, 3)
ax.axis('off')
plt.title("EfficientNet-B0 Architecture (Academic Style)", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()
