import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_bert_model():
    fig, ax = plt.subplots(figsize=(10, 12))

    def draw_box(ax, xy, width, height, label, boxstyle="round,pad=0.3", color="lightblue"):
        bbox = FancyBboxPatch(xy, width, height, boxstyle=boxstyle, ec="black", fc=color)
        ax.add_patch(bbox)
        ax.text(xy[0] + width/2, xy[1] + height/2, label, va="center", ha="center", fontsize=10, weight="bold")

    # Dimensions
    layer_width = 2.5
    layer_height = 1.0

    # Input Embedding
    draw_box(ax, [2, 11], layer_width, layer_height, "Token Embeddings", color="lightgreen")
    draw_box(ax, [5, 11], layer_width, layer_height, "Segment Embeddings", color="lightgreen")
    draw_box(ax, [8, 11], layer_width, layer_height, "Position Embeddings", color="lightgreen")
    draw_box(ax, [5, 10], layer_width, layer_height, "Input Embedding")

    # Draw arrows
    ax.plot([3.25, 5], [10.5, 10.5], color="black", lw=1.5)
    ax.plot([6.75, 5], [10.5, 10.5], color="black", lw=1.5)
    ax.plot([8.75, 5], [10.5, 10.5], color="black", lw=1.5)

    # Encoder Layers (simplified)
    for i in range(12):  # BERT-Base has 12 layers, BERT-Large has 24
        y_offset = 9 - i * 0.7
        draw_box(ax, [2, y_offset], layer_width, layer_height, f"Layer {i+1}", color="lightblue")
        draw_box(ax, [5, y_offset], layer_width, layer_height, "Multi-Head\nSelf-Attention", color="lightcoral")
        draw_box(ax, [8, y_offset], layer_width, layer_height, "Feed Forward", color="lightcoral")

        # Draw arrows between layers
        ax.plot([3.25, 5], [y_offset + 0.5, y_offset + 0.5], color="black", lw=1.5)
        ax.plot([6.75, 5], [y_offset + 0.5, y_offset + 0.5], color="black", lw=1.5)
        if i > 0:
            ax.plot([5, 5], [y_offset + 1.0, y_offset + 0.7], color="black", lw=1.5)
            ax.plot([8, 8], [y_offset + 1.0, y_offset + 0.7], color="black", lw=1.5)

    # Output layer
    draw_box(ax, [5, -0.5], layer_width, layer_height, "Output Layer", color="lightblue")
    ax.plot([5, 5], [0.5, -0.5], color="black", lw=1.5)

    # Set limits and turn off axis
    ax.set_xlim(0, 12)
    ax.set_ylim(-2, 12)
    ax.axis('off')

    plt.title("BERT Model Architecture", fontsize=16)
    plt.show()

draw_bert_model()
