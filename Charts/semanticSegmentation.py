import matplotlib.pyplot as plt
import numpy as np

# Data definitions
models = ['PIDNet\n(Apple M4 MPS)', 'SeaFormer++\n(RTX 3050 CUDA)']
miou = [78.21, 77.54]
fps = [44.41, 16.96]

x = np.arange(len(models))
width = 0.45

# Set up the plot
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

# Primary Axis: mIoU (Bars)
# Modern Steel Blue palette
color_miou = '#3F88C5' 
text_miou = '#1A4368' # Darker blue for readable text
bars = ax1.bar(x, miou, width, color=color_miou, alpha=0.85, edgecolor='#1A4368', linewidth=1.5)
ax1.set_ylabel('Accuracy (mIoU %) [Higher is Better]', fontweight='bold', color=text_miou, fontsize=11)
ax1.set_ylim(70, 82) # Narrowed Y-axis to highlight the precision difference
ax1.tick_params(axis='y', labelcolor=text_miou)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight='bold', fontsize=11, color='#333333')

# Annotate the mIoU bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f'{yval}%', 
             ha='center', va='bottom', fontweight='bold', color=text_miou)

# Secondary Axis: FPS (Line/Markers)
ax2 = ax1.twinx()
# Vibrant Coral/Orange palette
color_fps = '#F3722C' 
text_fps = '#B84A14' # Darker rust/orange for readable text
ax2.plot(x, fps, color=color_fps, marker='D', markersize=10, linewidth=3, linestyle='--', alpha=0.95)
ax2.set_ylabel('Inference Speed (FPS) [Higher is Better]', fontweight='bold', color=text_fps, fontsize=11)
ax2.set_ylim(0, 55)
ax2.tick_params(axis='y', labelcolor=text_fps)

# Annotate the FPS markers
for i, txt in enumerate(fps):
    # Offset the text slightly above the marker
    ax2.text(x[i], fps[i] + 2.5, f'{txt} FPS', 
             ha='center', va='bottom', fontweight='bold', color=text_fps)

# Formatting
plt.title('Semantic Segmentation: Accuracy vs. Inference Speed', fontsize=13, fontweight='bold', pad=15, color='#222222')
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# Hide top spine for a cleaner look
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Align everything tightly
fig.tight_layout()

# Save the figure
plt.savefig('sem_seg_dual_axis_modern.png', format='png', bbox_inches='tight')
plt.show()