import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_cnn_model():
    fig, ax = plt.subplots(figsize=(12, 10))

    def draw_box(ax, xy, width, height, label, boxstyle="round,pad=0.3", color="lightgreen"):
        bbox = FancyBboxPatch(xy, width, height, boxstyle=boxstyle, ec="black", fc=color)
        ax.add_patch(bbox)
        ax.text(xy[0] + width / 2, xy[1] + height / 2, label, va="center", ha="center", fontsize=12, weight="bold")

    layer_width = 2.5
    layer_height = 1.0

    # Input layer
    draw_box(ax, [1, 9], layer_width, layer_height, "Input Layer")

    # Layer 1: Conv1 -> BN1 -> ReLU -> MaxPool1
    draw_box(ax, [4, 9], layer_width, layer_height, "Conv1 (32)")
    draw_box(ax, [7, 9], layer_width, layer_height, "BN1")
    draw_box(ax, [10, 9], layer_width, layer_height, "ReLU1")
    draw_box(ax, [13, 9], layer_width, layer_height, "MaxPool1")

    # Layer 2: Conv2 -> BN2 -> ReLU -> MaxPool2
    draw_box(ax, [4, 7], layer_width, layer_height, "Conv2 (64)")
    draw_box(ax, [7, 7], layer_width, layer_height, "BN2")
    draw_box(ax, [10, 7], layer_width, layer_height, "ReLU2")
    draw_box(ax, [13, 7], layer_width, layer_height, "MaxPool2")

    # Layer 3: Conv3 -> BN3 -> ReLU -> MaxPool3
    draw_box(ax, [4, 5], layer_width, layer_height, "Conv3 (128)")
    draw_box(ax, [7, 5], layer_width, layer_height, "BN3")
    draw_box(ax, [10, 5], layer_width, layer_height, "ReLU3")
    draw_box(ax, [13, 5], layer_width, layer_height, "MaxPool3")

    # Fully Connected Layer 1
    draw_box(ax, [7, 3], layer_width, layer_height, "Flatten")
    draw_box(ax, [10, 3], layer_width, layer_height, "FC1 (256)")
    draw_box(ax, [13, 3], layer_width, layer_height, "Dropout")

    # Output Layer
    draw_box(ax, [10, 1], layer_width, layer_height, "Output Layer")

    # Connecting lines
    ax.plot([3.5, 4], [9.5, 9.5], color="black")
    ax.plot([6.5, 7], [9.5, 9.5], color="black")
    ax.plot([9.5, 10], [9.5, 9.5], color="black")
    ax.plot([12.5, 13], [9.5, 9.5], color="black")

    ax.plot([3.5, 4], [7.5, 7.5], color="black")
    ax.plot([6.5, 7], [7.5, 7.5], color="black")
    ax.plot([9.5, 10], [7.5, 7.5], color="black")
    ax.plot([12.5, 13], [7.5, 7.5], color="black")

    ax.plot([3.5, 4], [5.5, 5.5], color="black")
    ax.plot([6.5, 7], [5.5, 5.5], color="black")
    ax.plot([9.5, 10], [5.5, 5.5], color="black")
    ax.plot([12.5, 13], [5.5, 5.5], color="black")

    ax.plot([8.5, 7], [4.5, 3.5], color="black")
    ax.plot([9.5, 10], [3.5, 3.5], color="black")
    ax.plot([12.5, 13], [3.5, 3.5], color="black")

    ax.plot([11.5, 10], [2.5, 1.5], color="black")

    ax.set_xlim(0, 17)
    ax.set_ylim(0, 11)
    ax.axis('off')

    plt.title("Architecture of the CNN Model", fontsize=16)
    plt.show()


draw_cnn_model()
