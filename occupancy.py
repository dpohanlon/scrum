import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

import numpy as np

import pandas as pd
import seaborn as sns

from matplotlib import rcParams
import networkx as nx

from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde

from matplotlib.colors import LinearSegmentedColormap

import json

from model import sample_passenger_locations_per_station

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams['axes.linewidth'] = 2.0

rcParams.update({"figure.autolayout": True})

plt.rcParams["figure.figsize"] = (6, 6)

# Dictionary of London Underground lines with their corresponding hex colors
LONDON_UNDERGROUND_COLORS = {
    "Bakerloo": "#B36305",
    "Central": "#E32017",
    "Circle": "#FFD300",
    "District": "#00782A",
    "Hammersmith & City": "#F3A9BB",
    "Jubilee": "#A0A5A9",
    "Metropolitan": "#9B0056",
    "Northern": "#000000",
    "Piccadilly": "#003688",
    "Victoria": "#0098D4",
    "Waterloo & City": "#95CDBA",
    "Elizabeth": "#A0A5A9",  # Note: Shares color with Jubilee on some maps
}

def hex_to_colormap(hex_color, name="custom_colormap", n=256):
    """
    Creates a linear segmented colormap from a single hex color.

    Parameters:
        hex_color (str): The hex color code (e.g., '#E32017').
        name (str): The name of the colormap.
        n (int): Number of discrete colors in the colormap.

    Returns:
        LinearSegmentedColormap: A Matplotlib colormap object.
    """
    # Convert hex to RGB (values between 0 and 1)
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

    # Define a gradient from white to the specified color
    colors = [(1, 1, 1), rgb]  # Start with white, end with the line color
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n)
    return cmap

def get_boarders_alighters(file_name, line = 'PIC'):

    boarders_df = pd.read_excel(file_name, sheet_name = 'Station_Boarders')
    alighters_df = pd.read_excel(file_name, sheet_name = 'Station_Alighters')

    boarders_df.columns = boarders_df.iloc[1]
    boarders_df = boarders_df[2:]

    alighters_df.columns = alighters_df.iloc[1]
    alighters_df = alighters_df[2:]

    boarders_df = boarders_df[boarders_df['Line'] == 'PIC']
    alighters_df = alighters_df[alighters_df['Line'] == 'PIC']

    return boarders_df, alighters_df

def get_edge_direction(line, current_station, direction):
    """
    Maps the station and direction to the appropriate edge in the graph.
    Assumes 'forward' and 'reverse' as possible directions.
    Modify this function based on the actual 'Dir' values in your data.
    """

    # Define the main forward path
    if direction in ['WB', 'SB']:
        # Find the next station in the forward direction
        if current_station in line:
            idx = line.index(current_station)
            if idx < len(line) - 1:
                next_station = line[idx + 1]
                return (current_station, next_station)
    elif direction in ['EB', 'NB']:
        # Find the previous station in the reverse direction
        if current_station in line:
            idx = line.index(current_station)
            if idx > 0:
                prev_station = line[idx - 1]
                return (current_station, prev_station)
    return None  # Undefined direction

def create_graph(line_entrances):

    G = nx.DiGraph()

    for i in range(len(line_entrances) - 1):
        G.add_edge(line_entrances[i], line_entrances[i + 1], direction='forward')
        G.add_edge(line_entrances[i + 1], line_entrances[i], direction='reverse')

    return G

def populate_boarders_alighters_graph(G, line, boarders_df, alighters_df):

    # Initialize edge attributes
    for u, v in G.edges():
        G[u][v]['boarders'] = 0
        G[u][v]['alighters'] = 0

    # Process boarders data
    for index, row in boarders_df.iterrows():
        station = row['Station']
        direction = row['Dir']
        total = row['Total']

        edge = get_edge_direction(line, station, direction)
        if edge and G.has_edge(*edge):
            G[edge[0]][edge[1]]['boarders'] += total
        else:
            print(f"Boarders: Edge not found for Station: {station}, Direction: {direction}")

    # Process alighters data
    for index, row in alighters_df.iterrows():
        station = row['Station']
        direction = row['Dir']
        total = row['Total']

        edge = get_edge_direction(line, station, direction)
        if edge and G.has_edge(*edge):
            G[edge[0]][edge[1]]['alighters'] += total
        else:
            print(f"Alighters: Edge not found for Station: {station}, Direction: {direction}")

    # Create labels for edges based on boarders and alighters
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        label = f"B:{data['boarders']} A:{data['alighters']}"
        edge_labels[(u, v)] = label

    return G

def get_boarders_alighters_path(G, line, start, end):
    """
    Returns the path from start to end along with boarders and alighters for each segment.

    Parameters:
    - G (DiGraph): The directed graph of the Piccadilly Line.
    - start (str): Starting station.
    - end (str): Ending station.

    Returns:
    - path (list): List of stations from start to end.
    - boarders_list (list): List of boarders for each segment.
    - alighters_list (list): List of alighters for each segment.
    """

    try:
        path = nx.shortest_path(G, source=start, target=end)

        boarders_list = []
        alighters_list = []

        for i in range(len(path) - 1):
            origin = path[i]
            destination = path[i + 1]

            if G.has_edge(origin, destination):
                boarders = G[origin][destination].get('boarders', 0)
                alighters = G[origin][destination].get('alighters', 0)
            else:
                boarders = 0
                alighters = 0
                print(f"Warning: No edge from {origin} to {destination}")

            # A quick hack to ensure that we're going the same way as the line, otherwise boarders are alighters
            if line.index(start) > line.index(end):
                boarders, alighters = alighters, boarders

            boarders_list.append(boarders)
            alighters_list.append(alighters)

        return path, boarders_list, alighters_list

    except nx.NetworkXNoPath:
        print(f"No path found between {start} and {end}.")
        return None, None, None
    except nx.NodeNotFound as e:
        print(e)
        return None, None, None

def plot_passenger_locations_heatmap(passenger_locations_per_station, path):

    N = len(passenger_locations_per_station.values())  # Number of arrays
    np.random.seed(0)  # For reproducibility

    # Create a list of N arrays with random data between 0 and 100
    data_list = passenger_locations_per_station.values()

    # Define the number of bins and the range
    num_bins = 50
    range_min, range_max = 0, 100
    bins = np.linspace(range_min, range_max, num_bins + 1)  # 101 edges for 100 bins

    # Initialize a 2D array to hold histogram counts
    hist_matrix = np.zeros((N, num_bins))

    # Compute histograms for each array
    for i, data in enumerate(data_list):
        counts, _ = np.histogram(data, bins=bins)
        hist_matrix[i, :] = counts

    # Optionally, normalize the histograms (e.g., to frequency or probability)
    # Uncomment the following lines if normalization is desired
    hist_matrix = hist_matrix / hist_matrix.sum(axis=1, keepdims=True)

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))

    sns.heatmap(hist_matrix, cmap='viridis',
                cbar_kws={'label': 'Density'},
                xticklabels=False,
                yticklabels=path[:-1])  # Hide y-axis labels if N is large

    plt.title(f'Train occupancy')
    plt.ylabel('Station')
    plt.tight_layout()
    plt.savefig('/Users/dan/Downloads/picc_occupancy.pdf')
    plt.savefig('/Users/dan/Downloads/picc_occupancy.png')

def plot_occupancy_overlay(passenger_locations_per_station, station):

    # Path to your overlay image
    overlay_path = '/Users/dan/Downloads/trainsparency.png'
    overlay_img = Image.open(overlay_path).convert("RGBA")

    # Get image dimensions and aspect ratio
    img_width, img_height = overlay_img.size
    image_aspect = img_width / img_height

    # Convert the overlay image to a NumPy array
    overlay_np = np.array(overlay_img)

    # Define figure size based on image aspect ratio
    fig_height = 6  # inches
    fig_width = fig_height * image_aspect

    # Define DPI for the figure
    fig_dpi = 70

    # Create the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)

    # Assuming passenger_locations_per_station is a dictionary
    # Extract the passenger locations for the last station
    data = passenger_locations_per_station[station]

    # Define the number of bins for the histogram
    bins = 200
    bin_edges = np.linspace(min(data), max(data), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Compute histogram counts
    counts, _ = np.histogram(data, bins=bin_edges)

    # Perform Kernel Density Estimation for smoothing
    kde = gaussian_kde(data)
    kde_values = kde(bin_centers)

    # Normalize the KDE values for color mapping
    norm = Normalize(vmin=kde_values.min(), vmax=kde_values.max())
    # cmap = plt.cm.Blues
    cmap = hex_to_colormap(LONDON_UNDERGROUND_COLORS['Piccadilly'])
    colors = cmap(norm(kde_values))

    # Set a uniform height for all bars
    uniform_height = 1

    # Plot bars with fixed height and colors based on KDE values
    bars = ax.bar(bin_centers, [uniform_height]*bins, width=bin_width, color=colors, edgecolor='none', zorder=1)

    # Remove all axis ticks and labels for a clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

    # Hide all spines (borders) of the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Define the extent for the overlay image based on the histogram
    xmin, xmax = bin_edges[0], bin_edges[-1]
    ymin, ymax = 0, uniform_height

    extent = [xmin, xmax, ymin, ymax]

    # Overlay the image on top of the bars
    ax.imshow(
        overlay_np,
        extent=extent,
        aspect='auto',
        zorder=10,  # Ensure the image is on top
        alpha=1.0,  # Adjust transparency as needed
        interpolation='bilinear'  # Choose 'bilinear' or 'bicubic' for smoother images
    )

    # Adjust layout to fit everything nicely
    plt.tight_layout()

    # Define the output path for the final image
    output_path = '/Users/dan/Downloads/histogram_with_overlay_fixed_heights.png'

    # Save the figure with a transparent background
    fig.savefig(output_path, dpi=fig_dpi, bbox_inches='tight', transparent=True)

if __name__ == "__main__":

    # A dict from station names to entrance locations, (eastbound, westbound)
    line_entrances = json.load(open("line_entrances.json", "rb"))
    line = list(line_entrances["Piccadilly"][0].keys())

    print(line)

    picc_graph = create_graph(line) # Just one branch for now

    boarders_df, alighters_df = get_boarders_alighters('/Users/dan/Downloads/NBT23TWT_outputs.xlsx', line = 'PIC')

    picc_graph = populate_boarders_alighters_graph(picc_graph, line, boarders_df, alighters_df)

    path, boarders, alighters = get_boarders_alighters_path(picc_graph, line, "Cockfosters", "Earl's Court")

    print("\nPath:", " -> ".join(path))
    print("Boarders per segment:", boarders)
    print("Alighters per segment:", alighters)

    positions_on = [x[1] for x in line_entrances["Piccadilly"][0].values()]
    num_rounds = len(boarders)

    # Number of samples per station
    num_samples = 10000
    seed = 42

    # Sample passenger locations at each station
    passenger_locations_per_station, mixture_weights_per_station, alpha_samples = sample_passenger_locations_per_station(
        n_on=boarders,
        n_off=alighters,
        positions_on=positions_on,
        path = path,
        num_samples=num_samples,
        seed=seed
    )

    plot_passenger_locations_heatmap(passenger_locations_per_station, path)

    plot_occupancy_overlay(passenger_locations_per_station, 'Leicester Square')
