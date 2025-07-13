import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

import numpy as np

import pandas as pd

from matplotlib import rcParams

from tqdm import tqdm

import networkx as nx

import json

import re

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams["axes.linewidth"] = 2.0

rcParams.update({"figure.autolayout": True})

plt.rcParams["figure.figsize"] = (6, 6)


def create_london_underground_graph(df, separate_lines=True):
    """
    Creates NetworkX directed graph(s) from London Underground data.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the underground links.
    - separate_lines (bool): If True, returns a dictionary of graphs per line.
                             If False, returns a single global graph.

    Returns:
    - If separate_lines is False: A single NetworkX DiGraph.
    - If separate_lines is True: A dictionary where keys are line names and
      values are NetworkX DiGraph objects for each line.
    """
    if separate_lines:
        graphs = {}
        # Group the DataFrame by 'Line'
        grouped = df.groupby("Line")
        for line, group in grouped:
            G = nx.DiGraph(name=line)  # Ensure using DiGraph
            # Iterate over each row in the group and add edges
            for _, row in group.iterrows():
                from_station = row["From Station"]
                to_station = row["To Station"]
                # Add edge with attributes if needed
                G.add_edge(
                    from_station,
                    to_station,
                    Link=row["Link"],
                    Dir=row["Dir"],
                    Order=row["Order"],
                    From_NLC=row["From NLC"],
                    From_ASC=row["From ASC"],
                    To_NLC=row["To NLC"],
                    To_ASC=row["To ASC"],
                )
            graphs[line] = G
        return graphs
    else:
        G = nx.DiGraph(name="London Underground")  # Ensure using DiGraph
        # Iterate over each row and add edges
        for _, row in df.iterrows():
            from_station = row["From Station"]
            to_station = row["To Station"]
            line = row["Line"]
            # Add edge with attributes
            if G.has_edge(from_station, to_station):
                # If the edge already exists, append the line to the 'Lines' attribute
                if "Lines" in G[from_station][to_station]:
                    G[from_station][to_station]["Lines"].add(line)
                else:
                    G[from_station][to_station]["Lines"] = {line}
            else:
                G.add_edge(
                    from_station,
                    to_station,
                    Lines={line},
                    Link=row["Link"],
                    Dir=row["Dir"],
                    Order=row["Order"],
                    From_NLC=row["From NLC"],
                    From_ASC=row["From ASC"],
                    To_NLC=row["To NLC"],
                    To_ASC=row["To ASC"],
                )
        return G


def _is_subsequence(sub, main):
    if len(sub) > len(main):
        return False
    for i in range(len(main) - len(sub) + 1):
        if main[i : i + len(sub)] == sub:
            return True
    return False


def extract_station_sequences_from_graphs(line_graphs):
    """
    Extracts ordered station sequences for each line from NetworkX graphs
    by performing a DFS from every starting node.

    Parameters:
    - line_graphs (dict): Dictionary of NetworkX DiGraph objects per line.

    Returns:
    - dict: A dictionary with line names as keys and lists of complete station
      sequences (branches) as values.
    """
    lines_dict = {}

    for line, G in line_graphs.items():
        # Start stations have an in-degree of 0 or 1 (for loops/depots)
        start_nodes = [node for node, in_degree in G.in_degree() if in_degree <= 1]

        if not start_nodes:
            print(f"Warning: No start nodes found for line '{line}'.")
            continue

        all_paths = []

        for start_node in start_nodes:
            # Stack for DFS: stores (current_node, path_so_far)
            stack = [(start_node, [start_node])]

            while stack:
                current_node, path = stack.pop()

                # Successors are nodes connected by an outgoing edge
                successors = list(G.successors(current_node))

                # If there are no successors, we've reached a terminus (end of a branch)
                if not successors:
                    # Only add the path if it's not a trivial single-station path
                    if len(path) > 1:
                        all_paths.append(path)
                    continue

                # Explore each successor
                for next_node in successors:
                    # Avoid cycles in the path
                    if next_node not in path:
                        new_path = path + [next_node]
                        stack.append((next_node, new_path))
                    else:
                        # If a cycle is detected, treat the current path as complete
                        if len(path) > 1:
                            all_paths.append(path)

        # Remove duplicate paths that might have been found from different start points
        unique_paths = []
        seen_paths = set()
        for path in sorted(all_paths, key=len, reverse=True):
            path_tuple = tuple(path)
            # Ensure no sub-path is added if a longer path containing it already exists
            if not any(
                tuple(path) in seen
                for seen in seen_paths
                if len(seen) > len(path_tuple)
            ):
                if path_tuple not in seen_paths:
                    unique_paths.append(path)
                    seen_paths.add(path_tuple)

        lines_dict[line] = unique_paths

    return lines_dict


def save_sequences_to_json(sequences_dict, output_path):
    """
    Saves the line sequences dictionary to a JSON file.

    Parameters:
    - sequences_dict (dict): Dictionary with line names as keys and station sequences as values.
    - output_path (str): Path to the output JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(sequences_dict, json_file, ensure_ascii=False, indent=4)
    print(f"Line sequences have been saved to '{output_path}'.")


if __name__ == "__main__":

    excel_file_path = "/Users/dan/Downloads/NBT23TWT_outputs.xlsx"

    df = pd.read_excel(excel_file_path, sheet_name="Link_Frequencies")

    df.columns = df.iloc[1]
    df = df[2:]

    print(df.head())

    # Create separate graphs for each line
    line_graphs = create_london_underground_graph(df, separate_lines=True)

    # Example: Accessing the Bakerloo line graph
    bakerloo_graph = line_graphs.get("Bakerloo")

    # Display information about the Bakerloo graph
    print(
        f"Bakerloo Line has {bakerloo_graph.number_of_nodes()} stations and {bakerloo_graph.number_of_edges()} connections."
    )

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(bakerloo_graph, k=0.15, iterations=20)
    nx.draw(
        bakerloo_graph,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=100,
        font_size=10,
    )
    plt.savefig("bakerloo_line.png")

    line_sequences = extract_station_sequences_from_graphs(line_graphs)

    # Display the extracted sequences (optional)
    for line, sequences in line_sequences.items():
        print(f"Line: {line}")
        for seq in sequences:
            print(" -> ".join(seq))
        print("\n")

    # Step 4: Save the sequences to a JSON file
    output_json_path = (
        "london_underground_lines.json"  # Specify your desired output path
    )
    save_sequences_to_json(line_sequences, output_json_path)
