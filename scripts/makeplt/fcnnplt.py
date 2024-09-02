import matplotlib.pyplot as plt
import networkx as nx

def draw_fcnn():
    # Create a directed graph
    G = nx.DiGraph()

    # Define nodes (layers)
    layers = {
        "Input": (0, 5),
        "FC1 (1024)": (1, 5),
        "BN1": (2, 5),
        "ReLU1": (3, 5),
        "Dropout1": (4, 5),
        "FC2 (512)": (5, 5),
        "BN2": (6, 5),
        "ReLU2": (7, 5),
        "Dropout2": (8, 5),
        "FC3 (256)": (9, 5),
        "BN3": (10, 5),
        "ReLU3": (11, 5),
        "Dropout3": (12, 5),
        "Output": (13, 5),
    }

    # Add nodes to the graph
    for layer, pos in layers.items():
        G.add_node(layer, pos=pos)

    # Define edges (connections)
    edges = [
        ("Input", "FC1 (1024)"),
        ("FC1 (1024)", "BN1"),
        ("BN1", "ReLU1"),
        ("ReLU1", "Dropout1"),
        ("Dropout1", "FC2 (512)"),
        ("FC2 (512)", "BN2"),
        ("BN2", "ReLU2"),
        ("ReLU2", "Dropout2"),
        ("Dropout2", "FC3 (256)"),
        ("FC3 (256)", "BN3"),
        ("BN3", "ReLU3"),
        ("ReLU3", "Dropout3"),
        ("Dropout3", "Output"),
    ]

    # Add edges to the graph
    G.add_edges_from(edges)

    # Define positions for the nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20, edge_color="black")

    plt.title("FCNN Model Architecture", fontsize=16)
    plt.show()

draw_fcnn()
