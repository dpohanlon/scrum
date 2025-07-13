import matplotlib as mpl
import matplotlib.pyplot as plt

from pprint import pprint

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

rcParams["axes.linewidth"] = 2.0

rcParams.update({"figure.autolayout": True})

plt.rcParams["figure.figsize"] = (6, 6)

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
    "Elizabeth": "#A0A5A9",
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

    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    colors = [(1, 1, 1), rgb]
    cmap = LinearSegmentedColormap.from_list(name, colors, N=n)
    return cmap


def get_boarders_alighters(file_name, line="PIC"):

    boarders_df = pd.read_excel(file_name, sheet_name="Station_Boarders")
    alighters_df = pd.read_excel(file_name, sheet_name="Station_Alighters")

    boarders_df.columns = boarders_df.iloc[1]
    boarders_df = boarders_df[2:]

    alighters_df.columns = alighters_df.iloc[1]
    alighters_df = alighters_df[2:]

    boarders_df = boarders_df[boarders_df["Line"] == "PIC"]
    alighters_df = alighters_df[alighters_df["Line"] == "PIC"]

    return boarders_df, alighters_df


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
                boarders = G[origin][destination].get("boarders", 0)
                alighters = G[origin][destination].get("alighters", 0)
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

    N = len(passenger_locations_per_station.values())
    np.random.seed(0)

    data_list = passenger_locations_per_station.values()

    num_bins = 50
    range_min, range_max = 0, 100
    bins = np.linspace(range_min, range_max, num_bins + 1)

    hist_matrix = np.zeros((N, num_bins))

    for i, data in enumerate(data_list):
        counts, _ = np.histogram(data, bins=bins)
        hist_matrix[i, :] = counts

    hist_matrix = hist_matrix / hist_matrix.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        hist_matrix,
        cmap="viridis",
        cbar_kws={"label": "Density"},
        xticklabels=False,
        yticklabels=path[:-1],
    )

    plt.title(f"Train occupancy")
    plt.ylabel("Station")
    plt.tight_layout()
    plt.savefig("occupancy.png")


def plot_occupancy_overlay(passenger_locations_per_station, station):

    overlay_path = "assets/trainsparency.png"
    overlay_img = Image.open(overlay_path).convert("RGBA")

    img_width, img_height = overlay_img.size
    image_aspect = img_width / img_height

    overlay_np = np.array(overlay_img)

    fig_height = 6
    fig_width = fig_height * image_aspect

    fig_dpi = 70

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)

    data = passenger_locations_per_station[station]

    bins = 200
    bin_edges = np.linspace(min(data), max(data), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    counts, _ = np.histogram(data, bins=bin_edges)

    kde = gaussian_kde(data)
    kde_values = kde(bin_centers)

    norm = Normalize(vmin=kde_values.min(), vmax=kde_values.max())
    cmap = hex_to_colormap(LONDON_UNDERGROUND_COLORS["Piccadilly"])
    colors = cmap(norm(kde_values))

    uniform_height = 1

    bars = ax.bar(
        bin_centers,
        [uniform_height] * bins,
        width=bin_width,
        color=colors,
        edgecolor="none",
        zorder=1,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    xmin, xmax = bin_edges[0], bin_edges[-1]
    ymin, ymax = 0, uniform_height

    extent = [xmin, xmax, ymin, ymax]

    ax.imshow(
        overlay_np,
        extent=extent,
        aspect="auto",
        zorder=10,
        alpha=1.0,
        interpolation="bilinear",
    )

    plt.tight_layout()

    fig.savefig(
        "histogram_with_overlay.png", dpi=fig_dpi, bbox_inches="tight", transparent=True
    )


def make_plots(boarders_file, station, direction="SB", line="Piccadilly"):

    if direction.lower() in ["sb", "wb", "southbound", "westbound"]:
        direction = "SB"
    else:
        direction = "WB"

    if line.lower() != "piccadilly":
        print("Piccadilly Line only for now!")
        return


def create_line_graph(all_routes):
    """
    Builds one DiGraph covering every branch.
    Each edge carries a set dir_set = {'forward', 'reverse'} telling us which
    way(s) that physical track is ever travelled in the raw sequences.
    """

    def _add(G, u, v, tag):

        if G.has_edge(u, v):
            G[u][v].setdefault("dir_set", set()).add(tag)
        else:
            G.add_edge(
                u,
                v,
                direction=tag,  # keep one canonical tag
                dir_set={tag},  # **NEW**
                boarders=0,
                alighters=0,
            )

    G = nx.DiGraph()
    for seq in all_routes:
        for i in range(len(seq) - 1):
            print(seq[i], seq[i + 1], "forward")
            print(seq[i + 1], seq[i], "reverse")
            _add(G, seq[i], seq[i + 1], "forward")
            _add(G, seq[i + 1], seq[i], "reverse")
    return G


def populate_boarders_alighters_graph(G, boarders_df, alighters_df):
    """
    Distributes the per-station boarders/alighters data to matching edges
    (station -> next_station) or (prev_station -> station) that share the
    correct direction ('forward' or 'reverse'), based on your Dir codes.
    """

    # Assign boarders
    for _, row in boarders_df.iterrows():
        station = row["Station"]
        dir_code = row["Dir"]
        total = row["Total"]
        target_direction = code_to_direction(dir_code)

        if station not in G:
            print(f"Station {station} not in graph (boarders)")
            continue

        # Boarders go onto edges that start at this station, matching target_direction
        out_edges = []
        for nxt in G[station]:
            if target_direction in G[station][nxt]["dir_set"]:  # <-- was == before
                out_edges.append((station, nxt))

        if not out_edges:
            print(
                f"Boarders: No '{target_direction}' edges from {station} (dir_code {dir_code})"
            )
            continue

        share = total / len(out_edges)
        for u, v in out_edges:
            G[u][v]["boarders"] += share

    # Assign alighters
    for _, row in alighters_df.iterrows():
        station = row["Station"]
        dir_code = row["Dir"]
        total = row["Total"]
        target_direction = code_to_direction(dir_code)

        if station not in G:
            print(f"Station {station} not in graph (alighters)")
            continue

        # Alighters arrive on edges that end at this station, matching target_direction
        in_edges = []
        for pred in G.predecessors(station):
            if target_direction in G[pred][station]["dir_set"]:  # <-- was == before
                in_edges.append((pred, station))

        if not in_edges:
            print(
                f"Alighters: No '{target_direction}' edges into {station} (dir_code {dir_code})"
            )
            continue

        share = total / len(in_edges)
        for u, v in in_edges:
            G[u][v]["alighters"] += share

    return G


def code_to_direction(dir_code):
    # WB & SB trains travel “reverse” w.r.t. the JSON sequences,
    # EB & NB are “forward”.
    return "reverse" if dir_code in ["WB", "SB"] else "forward"


def get_all_paths_in_direction(G, start_station, termini, dir_code="WB"):
    """
    Return every simple path that a train reaching `start_station` could
    have travelled, restricted to edges usable by a train whose physical
    direction is given by `dir_code`.

    Parameters
    ----------
    G : nx.DiGraph          # network with 'direction' / 'dir_set' per edge
    start_station : str
    dir_code : {'WB','EB','SB','NB'}   # West/East/South/North-bound

    The helper `code_to_direction(dir_code)` must map the LU traffic code
    to the canonical edge tag ('forward' or 'reverse').
    """
    allowed_tag = code_to_direction(dir_code)  # ← 'forward' | 'reverse'

    # keep only edges whose dir_set contains the allowed tag
    H = nx.DiGraph(
        (
            (u, v, d)
            for u, v, d in G.edges(data=True)
            if allowed_tag in d.get("dir_set", {d["direction"]})
        )
    )

    # enumerate every simple path from the query station to each terminus
    all_paths = []
    for t in termini:
        all_paths.extend(nx.all_simple_paths(H, source=start_station, target=t))

    return all_paths


def merge_piccadilly_lists(line_entrances) -> list[str]:
    """
    Read the two Piccadilly-line station lists in *line_entrances.json*,
    build a precedence graph from every consecutive pair, and return one
    topological order that satisfies both original orderings.
    """

    seqs = [list(d.keys()) for d in line_entrances["Piccadilly"]]

    G = nx.DiGraph()
    for seq in seqs:  # add precedence edges
        for a, b in zip(seq[:-1], seq[1:]):
            G.add_edge(a, b)

    return list(nx.topological_sort(G))  # one ordering consistent with both


def load_piccadilly_routes(raw_routes, ordering):
    """
    Read line_entrances.json, force every Piccadilly-line route into a
    canonical west→east order, and drop the orientation duplicates.

    Returns
    -------
    list[list[str]]
        Each sub-list is a unique physical branch running west→east.
    """

    pos = {stn: i for i, stn in enumerate(ordering)}  # station → index

    def orient(r: list[str]) -> list[str]:
        """Flip route if its 2nd station lies further west than its 1st."""
        print(r[0], r[1], pos[r[1]] < pos[r[0]])
        return list(reversed(r)) if pos[r[1]] < pos[r[0]] else r

    oriented = [orient(r) for r in raw_routes]

    # dedupe by termini (keep first direction encountered)
    unique, seen = [], set()
    for r in oriented:
        termini = (r[0], r[-1])
        if (termini[1], termini[0]) not in seen:
            seen.add(termini)
            unique.append(r)

    return unique


if __name__ == "__main__":

    # A dict from station names to entrance locations, (eastbound, westbound)
    lines = json.load(open("london_underground_lines.json", "rb"))
    line_entrances = json.load(open("line_entrances.json", "rb"))

    routes = list(lines["Piccadilly"])

    # Looks weird, don't worry about it
    ordering = merge_piccadilly_lists(line_entrances)

    routes = load_piccadilly_routes(routes, ordering)

    route_graph = create_line_graph(routes)

    nx.draw(route_graph, with_labels=True)
    plt.savefig("graph.png")

    boarders_df, alighters_df = get_boarders_alighters(
        "/Users/dan/Downloads/NBT23TWT_outputs.xlsx", line="PIC"
    )

    route_graph = populate_boarders_alighters_graph(
        route_graph, boarders_df, alighters_df
    )

    termini = set([r[0] for r in routes] + [r[-1] for r in routes])

    forward_paths = get_all_paths_in_direction(
        route_graph, "Leicester Square", termini, dir_code="WB"
    )

    complete = [p for p in forward_paths if p[-1] in termini]
    branch_BA = [
        get_boarders_alighters_path(route_graph, p, p[0], p[-1]) for p in complete
    ]

    # Build a separate model for each 'route' and *then* average
    for i, (path, boarders, alighters) in enumerate(branch_BA):

        print(f"\nPath {i}:", " -> ".join(path))
        print("Boarders per segment:")
        pprint(list(zip(path, boarders)))
        print("Alighters per segment:")
        pprint(list(zip(path, alighters)))

        stations = dict(
            line_entrances["Piccadilly"][0], **line_entrances["Piccadilly"][1]
        )

        positions_on = [stations[s][1] for s in stations]

        num_samples = 1000
        seed = 42

        passenger_locations_per_station, mixture_weights_per_station, alpha_samples = (
            sample_passenger_locations_per_station(
                n_on=boarders,
                n_off=alighters,
                positions_on=positions_on,
                path=path,
                num_samples=num_samples,
                seed=seed,
            )
        )

        plot_passenger_locations_heatmap(passenger_locations_per_station, path)

        pprint(passenger_locations_per_station.keys())

        plot_occupancy_overlay(passenger_locations_per_station, "Piccadilly Circus")
