import pathlib
import sys
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
pd = pytest.importorskip('pandas')
nx = pytest.importorskip('networkx')

from make_network import (
    create_london_underground_graph,
    extract_station_sequences_from_graphs,
)


def sample_df():
    data = {
        'Line': ['Line1', 'Line1', 'Line1', 'Line1'],
        'From Station': ['A', 'B', 'B', 'C'],
        'To Station': ['B', 'C', 'D', 'D'],
        'Link': [1, 2, 3, 4],
        'Dir': ['0', '0', '0', '0'],
        'Order': [1, 2, 3, 4],
        'From NLC': [1, 2, 2, 3],
        'From ASC': [1, 2, 2, 3],
        'To NLC': [2, 3, 4, 4],
        'To ASC': [2, 3, 4, 4],
    }
    return pd.DataFrame(data)


def test_create_london_graph_global():
    df = sample_df()
    G = create_london_underground_graph(df, separate_lines=False)
    assert G.number_of_edges() == 4
    assert G.has_edge('A', 'B')
    assert G['A']['B']['Lines'] == {'Line1'}


def test_extract_station_sequences_from_graphs():
    df = sample_df()
    graphs = create_london_underground_graph(df, separate_lines=True)
    sequences = extract_station_sequences_from_graphs(graphs)
    expected = [['A', 'B', 'C', 'D'], ['A', 'B', 'D']]
    assert sorted(sequences['Line1']) == sorted(expected)


def branching_df():
    data = {
        'Line': ['Line1'] * 5,
        'From Station': ['A', 'B', 'C', 'B', 'D'],
        'To Station': ['B', 'C', 'E', 'D', 'E'],
        'Link': [1, 2, 3, 4, 5],
        'Dir': ['0'] * 5,
        'Order': [1, 2, 3, 4, 5],
        'From NLC': [1, 2, 3, 2, 4],
        'From ASC': [1, 2, 3, 2, 4],
        'To NLC': [2, 3, 5, 4, 5],
        'To ASC': [2, 3, 5, 4, 5],
    }
    return pd.DataFrame(data)


def test_extract_station_sequences_with_branching_merge():
    df = branching_df()
    graphs = create_london_underground_graph(df, separate_lines=True)
    sequences = extract_station_sequences_from_graphs(graphs)
    expected = [['A', 'B', 'C', 'E'], ['A', 'B', 'D', 'E']]
    assert sorted(sequences['Line1']) == sorted(expected)
