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

rcParams['axes.linewidth'] = 2.0

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
        grouped = df.groupby('Line')
        for line, group in grouped:
            G = nx.DiGraph(name=line)  # Ensure using DiGraph
            # Iterate over each row in the group and add edges
            for _, row in group.iterrows():
                from_station = row['From Station']
                to_station = row['To Station']
                # Add edge with attributes if needed
                G.add_edge(from_station, to_station,
                           Link=row['Link'],
                           Dir=row['Dir'],
                           Order=row['Order'],
                           From_NLC=row['From NLC'],
                           From_ASC=row['From ASC'],
                           To_NLC=row['To NLC'],
                           To_ASC=row['To ASC'])
            graphs[line] = G
        return graphs
    else:
        G = nx.DiGraph(name='London Underground')  # Ensure using DiGraph
        # Iterate over each row and add edges
        for _, row in df.iterrows():
            from_station = row['From Station']
            to_station = row['To Station']
            line = row['Line']
            # Add edge with attributes
            if G.has_edge(from_station, to_station):
                # If the edge already exists, append the line to the 'Lines' attribute
                if 'Lines' in G[from_station][to_station]:
                    G[from_station][to_station]['Lines'].add(line)
                else:
                    G[from_station][to_station]['Lines'] = {line}
            else:
                G.add_edge(from_station, to_station,
                           Lines={line},
                           Link=row['Link'],
                           Dir=row['Dir'],
                           Order=row['Order'],
                           From_NLC=row['From NLC'],
                           From_ASC=row['From ASC'],
                           To_NLC=row['To NLC'],
                           To_ASC=row['To ASC'])
        return G

def _is_subsequence(sub, main):
    if len(sub) > len(main):
        return False
    for i in range(len(main) - len(sub) + 1):
        if main[i:i+len(sub)] == sub:
            return True
    return False

def extract_station_sequences_from_graphs(line_graphs):
    """
    Extracts ordered station sequences for each line from NetworkX graphs,
    handling branches by duplicating common stations.

    Parameters:
    - line_graphs (dict): Dictionary where keys are line names and values are NetworkX DiGraph objects.

    Returns:
    - dict: A dictionary with line names as keys and lists of station sequences as values.
    """
    lines_dict = {}

    for line, G in line_graphs.items():
        # Identify start and end stations
        # Start stations have in-degree 1
        start_stations = [node for node, degree in G.in_degree() if degree == 1]
        # End stations have out-degree 1
        end_stations = [node for node, degree in G.out_degree() if degree == 1]

        if not start_stations:
            print(f"Warning: No start stations found for line '{line}'. Skipping.")
            continue
        if not end_stations:
            print(f"Warning: No end stations found for line '{line}'. Skipping.")
            continue

        # Initialize list to hold all sequences for the line
        sequences = []

        # For each start and end station, find all possible paths
        for start in start_stations:
            for end in end_stations:
                try:
                    # Find all simple paths from start to end
                    paths = list(nx.all_simple_paths(G, source=start, target=end))
                    sequences.extend(paths)
                except nx.NetworkXNoPath:
                    print(f"No path between {start} and {end} for line '{line}'.")

        # Remove duplicate sequences if any
        unique_sequences = []
        seen = set()
        for seq in sequences:
            seq_tuple = tuple(seq)
            if seq_tuple not in seen and len(seq_tuple) > 1:
                seen.add(seq_tuple)
                unique_sequences.append(seq)

        largest_sequences = []
        unique_sequences = sorted(unique_sequences, key=len, reverse=True)
        for i, seqA in enumerate(unique_sequences):
            # Only keep seqA if it's not fully contained in another longer sequence
            if not any(_is_subsequence(seqA, seqB) for j, seqB in enumerate(unique_sequences)
                       if j != i and len(seqB) >= len(seqA)):
                largest_sequences.append(seqA)

        lines_dict[line] = largest_sequences

    return lines_dict

def save_sequences_to_json(sequences_dict, output_path):
    """
    Saves the line sequences dictionary to a JSON file.

    Parameters:
    - sequences_dict (dict): Dictionary with line names as keys and station sequences as values.
    - output_path (str): Path to the output JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(sequences_dict, json_file, ensure_ascii=False, indent=4)
    print(f"Line sequences have been saved to '{output_path}'.")

if __name__ == "__main__":

    excel_file_path = '/Users/dan/Downloads/NBT23TWT_outputs.xlsx'

    df = pd.read_excel(excel_file_path, sheet_name = 'Link_Frequencies')

    df.columns = df.iloc[1]
    df = df[2:]

    print(df.head())

    # Create separate graphs for each line
    line_graphs = create_london_underground_graph(df, separate_lines=True)

    # Example: Accessing the Bakerloo line graph
    bakerloo_graph = line_graphs.get('Bakerloo')

    # Display information about the Bakerloo graph
    print(f"Bakerloo Line has {bakerloo_graph.number_of_nodes()} stations and {bakerloo_graph.number_of_edges()} connections.")

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(bakerloo_graph, k=0.15, iterations=20)
    nx.draw(bakerloo_graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=100, font_size=10)
    plt.savefig('bakerloo_line.png')

    line_sequences = extract_station_sequences_from_graphs(line_graphs)

    # Display the extracted sequences (optional)
    for line, sequences in line_sequences.items():
        print(f"Line: {line}")
        for seq in sequences:
            print(" -> ".join(seq))
        print("\n")

    # Step 4: Save the sequences to a JSON file
    output_json_path = 'london_underground_lines.json'  # Specify your desired output path
    save_sequences_to_json(line_sequences, output_json_path)
