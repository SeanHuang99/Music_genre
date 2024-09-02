import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_roberta_architecture():
    fig, ax = plt.subplots(figsize=(20, 8))

    def draw_box(ax, xy, width, height, label, boxstyle="round,pad=0.3", color="lightgreen"):
        bbox = FancyBboxPatch(xy, width, height, boxstyle=boxstyle, ec="black", fc=color)
        ax.add_patch(bbox)
        ax.text(xy[0] + width/2, xy[1] + height/2, label, va="center", ha="center", fontsize=10, weight="bold")

    # Define dimensions
    layer_width = 3.5  # Increased width for better text fitting
    layer_height = 1.5  # Increased height for better text fitting
    padding = 1.5  # Padding between layers

    # Input Embedding
    draw_box(ax, [1, 3], layer_width, layer_height, "Input Embedding\n(Token + Segment + Position)", color="lightblue")

    # Transformer Encoder Layers (simplified)
    for i in range(12):  # RoBERTa-Base has 12 layers
        x_offset = 1 + (i + 1) * (layer_width + padding)
        draw_box(ax, [x_offset, 3], layer_width, layer_height, f"Transformer\nEncoder Layer {i+1}", color="lightgreen")

    # Output Layer
    draw_box(ax, [1 + (12 + 1) * (layer_width + padding), 3], layer_width, layer_height, "Output Layer\n(Classification Head)", color="lightblue")

    # Connecting lines
    for i in range(12):
        x_start = 1 + (i + 1) * (layer_width + padding) - padding
        x_end = x_start + padding
        ax.plot([x_start, x_end], [4.25, 4.25], color="black", lw=1.5)

    ax.plot([1 + (12 + 1) * (layer_width + padding) - padding, 1 + (12 + 1) * (layer_width + padding)], [4.25, 4.25], color="black", lw=1.5)

    ax.set_xlim(0, 1 + (12 + 2) * (layer_width + padding))
    ax.set_ylim(1, 6)
    ax.axis('off')

    plt.title("RoBERTa Model Architecture", fontsize=16)
    plt.show()

draw_roberta_architecture()
