import matplotlib.pyplot as plt

# Data definitions
models = ['YOLO26n', 'RT-DETR']
latency_ms = [9.36, 70.41]
map_50_95 = [38.54, 63.46]
size_mb = [5.3, 66.2]

# Scale sizes for visual impact on the scatter plot
visual_sizes = [s * 30 for s in size_mb]

# Colors for contrast
colors = ['#1f77b4', '#ff7f0e']

# Set up the plot
plt.figure(figsize=(8, 6), dpi=300)
scatter = plt.scatter(latency_ms, map_50_95, s=visual_sizes, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

# Annotate the bubbles
for i, model in enumerate(models):
    plt.annotate(
        f"{model}\n({size_mb[i]} MB)",
        (latency_ms[i], map_50_95[i]),
        xytext=(15, -15) if i == 0 else (-15, 20),
        textcoords='offset points',
        ha='left' if i == 0 else 'right',
        fontsize=10,
        fontweight='bold'
    )

# Formatting the axes and grid
plt.title('Operational Trade-off: Accuracy vs. Latency vs. Memory Footprint', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Mean Latency (ms) [Lower is Better]', fontsize=11, fontweight='bold')
plt.ylabel('Accuracy (mAP@50-95 %) [Higher is Better]', fontsize=11, fontweight='bold')

# Set limits to create breathing room around the bubbles
plt.xlim(0, 90)
plt.ylim(30, 75)

# Add an "Ideal Operational Zone" marker
plt.axvspan(0, 30, color='green', alpha=0.1, label='Ideal Real-Time Zone (< 30ms)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right')
plt.tight_layout()

# Save the figure
plt.savefig('model_tradeoff_bubble.png', format='png', bbox_inches='tight')
plt.show()