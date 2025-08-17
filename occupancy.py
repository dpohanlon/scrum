import matplotlib as mpl
import matplotlib.pyplot as plt

from pprint import pprint

from scipy.stats import truncnorm

mpl.use("Agg")

from datetime import datetime

import numpy as np

import pandas as pd
import seaborn as sns

import h5py

import os

import re

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

def _prepare(df):
    df = df.copy()
    df.columns = df.iloc[1]           # row 1 becomes header
    df = df.iloc[2:].reset_index(drop=True)
    return df

def _sanitize_station_name(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")

def _round_to_half_hour(tstr):
    t = datetime.strptime(tstr, "%H:%M")
    m = (t.minute + 15) // 30 * 30
    if m == 60:
        t = (t.replace(minute=0) + timedelta(hours=1))
    else:
        t = t.replace(minute=m)
    return t.strftime("%H:%M")

def _times_30min():
    return [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)]

def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")  # vlen utf-8 strings

def _truncnorm_row(bin_centers, mean, std):
    a = (0 - mean) / std
    b = (100 - mean) / std
    vals = truncnorm.pdf(bin_centers, a, b, loc=mean, scale=std)
    s = vals.sum()
    return vals / s if s > 0 else vals

def live_relative_crowding(station, maxima_dict):
    return 1.0

def save_station_maxima_to_json(excel_path, json_path, line="PIC"):
    def _prepare(df):
        df = df.copy()
        df.columns = df.iloc[1]
        return df.iloc[2:].reset_index(drop=True)

    def _is_timerange(s):
        s = str(s).strip()
        return bool(re.fullmatch(r"\d{3,4}-\d{3,4}", s))

    b = _prepare(pd.read_excel(excel_path, sheet_name="Station_Boarders"))
    a = _prepare(pd.read_excel(excel_path, sheet_name="Station_Alighters"))

    b = b[b["Line"] == line]
    a = a[a["Line"] == line]

    time_cols_b = [c for c in b.columns if _is_timerange(c)]
    time_cols_a = [c for c in a.columns if _is_timerange(c)]
    time_cols = sorted(set(time_cols_b).intersection(time_cols_a))

    for c in time_cols:
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0)
        a[c] = pd.to_numeric(a[c], errors="coerce").fillna(0.0)

    maxima = {}
    for stn, g in b.groupby("Station"):
        bb = g[time_cols].sum(axis=0)
        aa = a[a["Station"] == stn][time_cols].sum(axis=0)
        tot = (bb + aa) if not aa.empty else bb
        maxima[stn] = float(tot.max()) if len(tot) else 0.0

    with open(json_path, "w") as f:
        json.dump(maxima, f, indent=2)

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


# def get_boarders_alighters(file_name, line="PIC"):

#     boarders_df = pd.read_excel(file_name, sheet_name="Station_Boarders")
#     alighters_df = pd.read_excel(file_name, sheet_name="Station_Alighters")

#     boarders_df.columns = boarders_df.iloc[1]
#     boarders_df = boarders_df[2:]

#     alighters_df.columns = alighters_df.iloc[1]
#     alighters_df = alighters_df[2:]

#     boarders_df = boarders_df[boarders_df["Line"] == "PIC"]
#     alighters_df = alighters_df[alighters_df["Line"] == "PIC"]

#     return boarders_df, alighters_df

def get_boarders_alighters(file_name, line="PIC", time_str=None, tolerance_min=30, boarders_df=None, alighters_df=None):
    """
    Read Excel sheets 'Station_Boarders' / 'Station_Alighters'.
    After promoting the 2nd row to headers and dropping the top two rows,
    filter to `line`. If `time_str='HH:MM'` is given, pick the nearest
    time-range column like '1400-1415' and expose it as 'Total'.

    If no time is given, the original 'Total' column is left as-is.
    """

    def _to_minutes(hhmm):
        h, m = map(int, hhmm.split(":"))
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError
        return h * 60 + m

    def _is_timerange(s):
        s = str(s).strip()
        return bool(re.fullmatch(r"\d{3,4}-\d{3,4}", s))

    def _midpoint_minutes(rng):
        a, b = str(rng).split("-")
        a = a.zfill(4); b = b.zfill(4)
        ah, am = int(a[:2]), int(a[2:])
        bh, bm = int(b[:2]), int(b[2:])
        t1, t2 = ah*60 + am, bh*60 + bm
        if t2 < t1:
            t2 += 24*60
        return ((t1 + t2) / 2.0) % (24*60)

    def _select_time_col(df, want_minutes):
        time_cols = [c for c in df.columns if _is_timerange(c)]
        if not time_cols:
            return None, None
        mids = np.array([_midpoint_minutes(c) for c in time_cols])
        idx  = int(np.argmin(np.abs(mids - want_minutes)))
        chosen = time_cols[idx]
        delta  = abs(int(round(mids[idx] - want_minutes)))
        return chosen, delta

    if boarders_df is None:
        boarders_df  = _prepare(pd.read_excel(file_name, sheet_name="Station_Boarders"))
    if alighters_df is None:
        alighters_df = _prepare(pd.read_excel(file_name, sheet_name="Station_Alighters"))

    # filter line (exact code, e.g. 'PIC')
    boarders_df  = boarders_df[boarders_df["Line"] == line]
    alighters_df = alighters_df[alighters_df["Line"] == line]

    if time_str is None:
        return boarders_df, alighters_df

    # parse time and select nearest range column in each sheet
    want = _to_minutes(time_str)
    b_col, b_delta = _select_time_col(boarders_df, want)
    a_col, a_delta = _select_time_col(alighters_df, want)

    if b_col is None or a_col is None:
        raise RuntimeError("No time-range columns like 'HHMM-HHMM' found in one or both sheets.")
    if (b_delta is not None and b_delta > tolerance_min) or (a_delta is not None and a_delta > tolerance_min):
        raise RuntimeError(
            f"No time column within {tolerance_min} min of {time_str}: "
            f"nearest boarders={b_col} (Δ≈{b_delta} min), alighters={a_col} (Δ≈{a_delta} min)."
        )

    # expose chosen time columns as numeric 'Total'
    boarders_df  = boarders_df.copy()
    alighters_df = alighters_df.copy()
    boarders_df["Total"]  = pd.to_numeric(boarders_df[b_col], errors="coerce").fillna(0.0)
    alighters_df["Total"] = pd.to_numeric(alighters_df[a_col], errors="coerce").fillna(0.0)

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


def plot_passenger_locations_heatmap(passenger_locations_per_station, path, name = None):

    station_order = [s for s in path if s in passenger_locations_per_station]
    N = len(station_order)
    np.random.seed(0)

    data_list = [passenger_locations_per_station[s] for s in station_order]

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
        yticklabels=station_order,
    )

    plt.title(f"Train occupancy")
    plt.ylabel("Station")
    plt.tight_layout()
    plt.savefig("occupancy.png" if name == None else name)

def generate_live_overlay(current_time, station, direction, hdf5_dir, maxima_json, out_png, bins=200, std=30, overlay_path="assets/trainsparency.png", line_key="Piccadilly"):
    tkey = _round_to_half_hour(current_time).replace(":", "")
    fn = os.path.join(hdf5_dir, f"{_sanitize_station_name(station)}.h5")
    maxima = json.load(open(maxima_json, "rb"))
    line_entrances = json.load(open("line_entrances.json", "rb"))
    stations_pos = dict(line_entrances[line_key][0], **line_entrances[line_key][1])
    pos_idx = 1 if direction in ("WB", "SB") else 0

    xs = np.linspace(0, 100, bins + 1)
    centers = 0.5 * (xs[:-1] + xs[1:])
    mix_total = np.zeros_like(centers, dtype=float)

    with h5py.File(fn, "r") as h5:
        routes_grp = h5[direction][tkey]["routes"]
        n_routes = len(routes_grp.keys())
        for key in sorted(routes_grp.keys(), key=lambda s: int(s[1:])):
            g = routes_grp[key]
            full_path = [s for s in g["stations"][...].astype(str)]
            pivot_idx = int(g.attrs["pivot_idx"])
            upstream = full_path[:pivot_idx + 1]

            ws = np.array([live_relative_crowding(s, maxima) for s in upstream], dtype=float)
            sw = ws.sum()
            ws = ws / sw if sw > 0 else np.ones(len(upstream), dtype=float) / max(1, len(upstream))

            mix_route = np.zeros_like(centers, dtype=float)
            for s, w in zip(upstream, ws):
                m = stations_pos[s][pos_idx]
                mix_route += w * _truncnorm_row(centers, m, std)

            mix_total += mix_route

    mix_total /= max(1, n_routes)

    overlay_img = Image.open(overlay_path).convert("RGBA")
    img_w, img_h = overlay_img.size
    aspect = img_w / img_h
    overlay_np = np.array(overlay_img)

    fig_h = 6
    fig_w = fig_h * aspect
    dpi = 70

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    norm = Normalize(vmin=mix_total.min(), vmax=mix_total.max())
    cmap = hex_to_colormap(LONDON_UNDERGROUND_COLORS.get(line_key, "#003688"))
    colors = cmap(norm(mix_total))

    bin_w = centers[1] - centers[0]
    ax.bar(centers, np.ones_like(centers), width=bin_w, color=colors, edgecolor="none", zorder=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for sp in ax.spines.values():
        sp.set_visible(False)

    extent = [0, 100, 0, 1]
    ax.imshow(overlay_np, extent=extent, aspect="auto", zorder=10, alpha=1.0, interpolation="bilinear")

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)

def plot_occupancy_overlay(passenger_locations_per_station, station, name = None):

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
        "histogram_with_overlay.png" if name == None else name, dpi=fig_dpi, bbox_inches="tight", transparent=True
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

def expected_onboard_per_station(n_on, n_off):
    """P_k: expected # on board before alighting at station k (snapshot)."""
    P = []
    load = 0.0
    for on, off in zip(n_on, n_off):
        P.append(load)                 # before alighting at this station
        off_eff = min(off, load)       # can’t alight more than are on
        load = load - off_eff + on     # after this station’s turn
    return np.array(P)                 # len = len(n_on)

def directional_full_paths(G, dir_code):
    """
    Return every full path from a direction's sources (no predecessors)
    to its sinks (no successors), using only edges usable in `dir_code`.
    """
    allowed = code_to_direction(dir_code)
    H = nx.DiGraph(
        (u, v, d) for u, v, d in G.edges(data=True)
        if allowed in d.get("dir_set", {d["direction"]})
    )
    sources = [n for n in H if H.in_degree(n) == 0]
    sinks   = [n for n in H if H.out_degree(n) == 0]

    paths = []
    for s in sources:
        for t in sinks:
            if s == t:
                continue
            try:
                for p in nx.all_simple_paths(H, source=s, target=t):
                    paths.append(p)
            except nx.NetworkXNoPath:
                pass
    return paths


def station_arrays_for_path(G, path):
    """
    Build station-aligned arrays along `path`:
    n_on[i]  = boarders at station path[i] (edge path[i]→path[i+1])
    n_off[i] = alighters at station path[i] (edge path[i-1]→path[i]); n_off[0]=0
    """
    n = len(path) - 1
    n_on  = [0.0] * n
    n_off = [0.0] * n
    for i, (u, v) in enumerate(zip(path[:-1], path[1:])):
        n_on[i] = G[u][v].get("boarders", 0.0)
        if i > 0:
            n_off[i] = G[path[i-1]][path[i]].get("alighters", 0.0)
    return n_on, n_off

def direction_subgraph(G, dir_code):
    """Keep only edges usable in this LU dir_code ('WB','EB','SB','NB')."""
    allowed = code_to_direction(dir_code)  # 'forward' | 'reverse'
    return nx.DiGraph(
        (u, v, d) for u, v, d in G.edges(data=True)
        if allowed in d.get("dir_set", {d["direction"]})
    )

def upstream_chain(H, pivot):
    """
    Follow unique predecessors from `pivot` until there are none.
    For Piccadilly WB this walks Cockfosters → … → pivot.
    """
    chain = [pivot]
    cur = pivot
    while True:
        preds = list(H.predecessors(cur))
        if not preds:                     # reached the upstream terminus
            break
        if len(preds) > 1:
            # If this ever happens on Piccadilly, pick one deterministically.
            # (Not expected WB; there is a single east-side branch.)
            preds.sort()
        cur = preds[0]
        chain.insert(0, cur)
    return chain

WEST_TERMINI = {"Uxbridge", "Heathrow Terminal 5 LU", "Heathrow Terminal 4 LU"}
EAST_TERMINI = {"Cockfosters"}

def direction_subgraph(G, dir_code):
    """Keep only edges usable in this LU dir_code."""
    allowed = code_to_direction(dir_code)  # 'forward' | 'reverse'
    return nx.DiGraph(
        (u, v, d) for u, v, d in G.edges(data=True)
        if allowed in d.get("dir_set", {d["direction"]})
    )

def build_paths_through_pivot(G, pivot, dir_code):
    """
    Return full paths that *pass through* `pivot`, built as:
      [upstream-terminus → … → pivot]  +  [pivot → … → target-terminus]
    using only edges consistent with `dir_code`.
    """
    H = direction_subgraph(G, dir_code)
    if pivot not in H:
        return []  # nothing to do

    if dir_code in ("WB", "SB"):
        sources = [t for t in EAST_TERMINI if t in H]   # upstream side
        targets = [t for t in WEST_TERMINI if t in H]
    else:  # EB/NB
        sources = [t for t in WEST_TERMINI if t in H]
        targets = [t for t in EAST_TERMINI if t in H]

    # pick the first upstream terminus that can reach the pivot
    src = next((s for s in sources if nx.has_path(H, s, pivot)), None)
    if src is None:
        return []

    prefix = nx.shortest_path(H, src, pivot)  # includes pivot at the end

    paths = []
    for t in targets:
        if nx.has_path(H, pivot, t):
            for tail in nx.all_simple_paths(H, source=pivot, target=t):
                paths.append(prefix[:-1] + tail)  # avoid double 'pivot'
    return paths

def get_paths(station, direction, time, boarders_file = "/Users/dan/Downloads/NBT23TWT_outputs.xlsx", boarders_df=None, alighters_df=None):

    lines = json.load(open("london_underground_lines.json", "rb"))
    line_entrances = json.load(open("line_entrances.json", "rb"))

    routes = list(lines["Piccadilly"])

    # Looks weird, don't worry about it
    ordering = merge_piccadilly_lists(line_entrances)

    routes = load_piccadilly_routes(routes, ordering)

    route_graph = create_line_graph(routes)

    boarders_df, alighters_df = get_boarders_alighters(
        boarders_file, line="PIC", time_str = time, boarders_df=boarders_df, alighters_df=alighters_df
    )

    route_graph = populate_boarders_alighters_graph(
        route_graph, boarders_df, alighters_df
    )

    paths_through_pivot = build_paths_through_pivot(route_graph, station, direction)

    # merged entrance dict for west/east positions
    stations = dict(line_entrances["Piccadilly"][0], **line_entrances["Piccadilly"][1])

    return paths_through_pivot, route_graph, stations

def get_occupancy(station = "Leicester Square", direction = "WB", time = "08:00"):

    paths_through_pivot, route_graph, stations = get_paths(station, direction, time)

    pos_idx  = 1 if direction in ["WB", "SB"] else 0   # westbound uses index 1

    num_samples = 1000
    seed = 42

    locs = []

    for i, full_path in enumerate(paths_through_pivot):
        # 2) station-based on/off along the *full* path from terminus
        n_on, n_off = station_arrays_for_path(route_graph, full_path)

        # 3) positions aligned to the full path
        positions_on = [stations[s][pos_idx] for s in full_path[:-1]]

        # 4) run the model on the full path (starts at the terminus → no fake initial load)
        passenger_locs, mix_wts, alpha_samples = sample_passenger_locations_per_station(
            n_on=n_on,
            n_off=n_off,
            positions_on=positions_on,
            path=full_path,
            num_samples=num_samples,
            seed=seed,
        )

        locs.append(passenger_locs)

    return locs

def get_occupancy_hist(this_station = "Leicester Square", direction = "WB", time = "08:00", plot = False):

    paths_through_pivot, route_graph, stations = get_paths(this_station, direction, time)

    counts = []

    for i, full_path in enumerate(paths_through_pivot):

        SCALE = 50                 # samples per person (reduce if memory is tight)
        STD   = 30                 # cm-ish; same as before
        idx   = 1 if direction in ("WB","SB") else 0   # westbound uses index 1

        n_on, n_off = station_arrays_for_path(route_graph, full_path)

        pivot_idx = full_path.index(this_station)

        passengers           = []        # evolving list of positions on board
        passengers_station   = []        # snapshots to plot (from pivot onward)
        station_labels       = []        # y-axis labels (from pivot onward)

        for j, station in enumerate(full_path[:-1]):
            print(i, j)
            # 1) alight FIRST at this station
            np.random.shuffle(passengers)
            off = int(n_off[j])
            if off > 0:
                passengers = passengers[:-SCALE * off] if SCALE * off <= len(passengers) else []

            # 2) then board at this station
            mean = stations[station][idx]
            a, b = (0 - mean) / STD, (100 - mean) / STD
            on   = int(n_on[j])
            if on > 0:
                boarding = truncnorm.rvs(a, b, loc=mean, scale=STD, size=SCALE * on)
                passengers.extend(boarding)

            # 3) record snapshot only from the pivot onward
            if j >= pivot_idx:
                passengers_station.append(np.array(passengers))
                station_labels.append(station)

        # ---------- expected-counts heatmap (rows scale with people aboard) ----------
        P_full = expected_onboard_per_station(n_on, n_off)      # along full path
        P_plot = P_full[pivot_idx:pivot_idx + len(station_labels)]

        counts.append(P_full)

        if plot:

            # ---------- probability heatmap (rows sum to 1) ----------
            num_bins = 100
            bins = np.linspace(0, 100, num_bins + 1)
            hist_prob = np.zeros((len(passengers_station), num_bins))
            for p, data in enumerate(passengers_station):
                counts, _ = np.histogram(data, bins=bins)
                s = counts.sum()
                hist_prob[p, :] = counts / s if s > 0 else counts

            plt.figure(figsize=(12, 8))
            sns.heatmap(hist_prob, cmap='viridis',
                        cbar_kws={'label': 'Probability'},
                        xticklabels=False, yticklabels=station_labels)
            plt.title('Train occupancy – probability per station')
            plt.ylabel('Station')
            plt.tight_layout()
            plt.savefig(f'occupancy_prob_{i}.png')

            hist_counts = np.zeros_like(hist_prob, dtype=float)
            for p, data in enumerate(passengers_station):
                counts, _ = np.histogram(data, bins=bins)
                s = counts.sum()
                row_pdf = counts / s if s > 0 else counts
                hist_counts[p, :] = row_pdf * P_plot[p]

            plt.figure(figsize=(12, 8))
            sns.heatmap(hist_counts, cmap='viridis',
                        cbar_kws={'label': 'People (expected)'},
                        xticklabels=False, yticklabels=station_labels)
            plt.title('Train occupancy – expected counts per station')
            plt.ylabel('Station')
            plt.tight_layout()
            plt.savefig(f'occupancy_counts_{i}.png')

    return counts

def save_data(boarders_file, output_dir, directions=("WB", "EB"), stations=None, times=None, std=30):
    os.makedirs(output_dir, exist_ok=True)
    line_entrances = json.load(open("line_entrances.json", "rb"))
    all_stations = set(line_entrances["Piccadilly"][0].keys()) | set(line_entrances["Piccadilly"][1].keys())
    if stations is None:
        stations = sorted(all_stations)
    if times is None:
        times = _times_30min()

    boarders_df  = _prepare(pd.read_excel(boarders_file, sheet_name="Station_Boarders"))
    alighters_df = _prepare(pd.read_excel(boarders_file, sheet_name="Station_Alighters"))

    for stn in stations:
        safe_stn = _sanitize_station_name(stn)
        fname = os.path.join(output_dir, f"{safe_stn}.h5")
        with h5py.File(fname, "w") as h5:
            h5.attrs["pivot_station"] = stn
            for direction in directions:
                dir_grp = h5.create_group(direction)
                for time_str in times:
                    print(direction, time_str, stn)
                    paths_through_pivot, route_graph, stations_pos = get_paths(
                        stn, direction, time_str, boarders_df=boarders_df, alighters_df=alighters_df
                    )
                    tgrp = dir_grp.create_group(time_str.replace(":", ""))
                    routes_grp = tgrp.create_group("routes")
                    for i, full_path in enumerate(paths_through_pivot):
                        n_on, n_off = station_arrays_for_path(route_graph, full_path)
                        p_full = expected_onboard_per_station(n_on, n_off)
                        g = routes_grp.create_group(f"r{i}")
                        g.create_dataset("stations", data=np.array(full_path, dtype=_string_dtype()))
                        g.create_dataset("counts", data=np.array(p_full, dtype=np.float64))
                        g.attrs["pivot_idx"] = int(full_path.index(stn))


def generate_live_histogram(current_time, station, direction, hdf5_dir, maxima_json, out_png, bins=100, std=30):
    tkey = _round_to_half_hour(current_time).replace(":", "")
    fn = os.path.join(hdf5_dir, f"{_sanitize_station_name(station)}.h5")
    maxima = json.load(open(maxima_json, "rb"))
    line_entrances = json.load(open("line_entrances.json", "rb"))
    stations_pos = dict(line_entrances["Piccadilly"][0], **line_entrances["Piccadilly"][1])
    pos_idx = 1 if direction in ("WB", "SB") else 0

    xs = np.linspace(0, 100, bins + 1)
    centers = 0.5 * (xs[:-1] + xs[1:])
    mix_total = np.zeros_like(centers, dtype=float)

    with h5py.File(fn, "r") as h5:
        routes_grp = h5[direction][tkey]["routes"]
        n_routes = len(routes_grp.keys())
        for key in sorted(routes_grp.keys(), key=lambda s: int(s[1:])):
            g = routes_grp[key]
            full_path = [s for s in g["stations"][...].astype(str)]
            pivot_idx = int(g.attrs["pivot_idx"])
            upstream = full_path[:pivot_idx + 1]

            ws = np.array([live_relative_crowding(s, maxima) for s in upstream], dtype=float)
            sw = ws.sum()
            if sw > 0:
                ws = ws / sw
            else:
                ws = np.ones(len(upstream), dtype=float) / max(1, len(upstream))

            mix_route = np.zeros_like(centers, dtype=float)
            for s, w in zip(upstream, ws):
                m = stations_pos[s][pos_idx]
                mix_route += w * _truncnorm_row(centers, m, std)

            mix_total += mix_route

    mix_total /= max(1, n_routes)
    plt.figure()
    bin_w = centers[1] - centers[0]
    plt.bar(centers, mix_total, width=bin_w, edgecolor="none")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, transparent=True)



def debug():

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
        "/Users/dan/Downloads/NBT23TWT_outputs.xlsx", line="PIC",
    )

    route_graph = populate_boarders_alighters_graph(
        route_graph, boarders_df, alighters_df
    )

    dir_code   = "WB"                     # westbound example
    pivot      = "Leicester Square"

    paths_through_pivot = build_paths_through_pivot(route_graph, pivot, dir_code)
    full_paths = paths_through_pivot
    print(f"Found {len(full_paths)} {dir_code} paths through {pivot}:")
    for p in full_paths:
        print(" -> ".join(p))

    # merged entrance dict for west/east positions
    stations = dict(line_entrances["Piccadilly"][0], **line_entrances["Piccadilly"][1])
    pos_idx  = 1 if dir_code in ["WB", "SB"] else 0   # westbound uses index 1

    print(full_paths)

    num_samples = 1000
    seed = 42

    for i, full_path in enumerate(paths_through_pivot):
        # 2) station-based on/off along the *full* path from terminus
        n_on, n_off = station_arrays_for_path(route_graph, full_path)

        # 3) positions aligned to the full path
        positions_on = [stations[s][pos_idx] for s in full_path[:-1]]

        # 4) run the model on the full path (starts at the terminus → no fake initial load)
        passenger_locs, mix_wts, alpha_samples = sample_passenger_locations_per_station(
            n_on=n_on,
            n_off=n_off,
            positions_on=positions_on,
            path=full_path,
            num_samples=num_samples,
            seed=seed,
        )

        # 5) slice outputs so we only plot from `pivot` onwards
        j = full_path.index(pivot)                # pivot row in the path
        sub_path = full_path[j:]                  # stations from pivot → terminus
        sub_loc_dict = {st: passenger_locs[st] for st in sub_path[1:]}  # keep order

        # plots
        plot_passenger_locations_heatmap(sub_loc_dict, sub_path, f"occupancy_stations_path_{i}")
        plot_occupancy_overlay(passenger_locs, "Covent Garden", f"occupancy_path_{i}")

        # print([x.shape for x in mixture_weights_per_station.values()])

        # print(np.sum(mixture_weights_per_station['Knightsbridge']))
        # print(mixture_weights_per_station['Knightsbridge'])

        # inputs already available in your debug(): full_path, n_on, n_off, stations, dir_code, pivot
        SCALE = 50                 # samples per person (reduce if memory is tight)
        STD   = 30                 # cm-ish; same as before
        idx   = 1 if dir_code in ("WB","SB") else 0   # westbound uses index 1

        pivot_idx = full_path.index(pivot)

        passengers           = []        # evolving list of positions on board
        passengers_station   = []        # snapshots to plot (from pivot onward)
        station_labels       = []        # y-axis labels (from pivot onward)

        for i, station in enumerate(full_path[:-1]):        # simulate along the entire path
            # 1) alight FIRST at this station
            np.random.shuffle(passengers)
            off = int(n_off[i])
            if off > 0:
                passengers = passengers[:-SCALE * off] if SCALE * off <= len(passengers) else []

            # 2) then board at this station
            mean = stations[station][idx]
            a, b = (0 - mean) / STD, (100 - mean) / STD
            on   = int(n_on[i])
            if on > 0:
                boarding = truncnorm.rvs(a, b, loc=mean, scale=STD, size=SCALE * on)
                passengers.extend(boarding)

            # 3) record snapshot only from the pivot onward
            if i >= pivot_idx:
                passengers_station.append(np.array(passengers))
                station_labels.append(station)

        # ---------- probability heatmap (rows sum to 1) ----------
        num_bins = 100
        bins = np.linspace(0, 100, num_bins + 1)
        hist_prob = np.zeros((len(passengers_station), num_bins))
        for i, data in enumerate(passengers_station):
            counts, _ = np.histogram(data, bins=bins)
            s = counts.sum()
            hist_prob[i, :] = counts / s if s > 0 else counts

        plt.figure(figsize=(12, 8))
        sns.heatmap(hist_prob, cmap='viridis',
                    cbar_kws={'label': 'Probability'},
                    xticklabels=False, yticklabels=station_labels)
        plt.title('Train occupancy – probability per station')
        plt.ylabel('Station')
        plt.tight_layout()
        plt.savefig('occupancy_prob.png')

        # ---------- expected-counts heatmap (rows scale with people aboard) ----------
        P_full = expected_onboard_per_station(n_on, n_off)      # along full path
        P_plot = P_full[pivot_idx:pivot_idx + len(station_labels)]

        hist_counts = np.zeros_like(hist_prob, dtype=float)
        for i, data in enumerate(passengers_station):
            counts, _ = np.histogram(data, bins=bins)
            s = counts.sum()
            row_pdf = counts / s if s > 0 else counts
            hist_counts[i, :] = row_pdf * P_plot[i]

        plt.figure(figsize=(12, 8))
        sns.heatmap(hist_counts, cmap='viridis',
                    cbar_kws={'label': 'People (expected)'},
                    xticklabels=False, yticklabels=station_labels)
        plt.title('Train occupancy – expected counts per station')
        plt.ylabel('Station')
        plt.tight_layout()
        plt.savefig('occupancy_counts.png')
        exit(0)

if __name__ == "__main__":
    # debug()
    # occ = get_occupancy()
    # print(occ.keys())
    # print(occ['Leicester Square'])

    # get_occupancy_hist(this_station = "Leicester Square", direction = "WB", time = "08:00")

    # save_station_maxima_to_json('/Users/dan/Downloads/NBT23TWT_outputs.xlsx', '')
    # save_data('/Users/dan/Downloads/NBT23TWT_outputs.xlsx', 'data')

    generate_live_histogram("09:35", "South Kensington", "WB", 'data', 'historical_maxima.json', 'hist.png', bins=100, std=30)

    generate_live_overlay("09:35", "South Kensington", "WB", 'data', 'historical_maxima.json', 'overlay.png', bins=200, std=30, overlay_path="assets/trainsparency.png", line_key="Piccadilly")
