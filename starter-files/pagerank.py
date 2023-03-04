"""
pagerank.py.

Implements the PageRank algorithm over a DirectedGraph.

This module can be invoked at the command line to compute the
PageRanks of the nodes in a digraph represented as two files, one with
nodes and the other with edges:

  python3 pagerank.py <node_file> <edge_file> [<num_iterations>]

The node and edge files must match the format defined in the spec.

To run the doctests in this module, use the following command:

  python3 -m doctest pagerank.py

You can also pass the -v flag to get more detailed feedback from the
tests.

Project UID bd3b06d8a60861e18088226c3a1f0595e4426dcf
"""

import sys
import graph
import numpy as np
from tqdm import tqdm


def pagerank(digraph, num_iterations=40, damping_factor=.85):
    """Calculate the PageRank for the nodes in the given digraph.

    In num_iterations iterations, calculates the PageRank for all
    nodes in the given digraph according to the formula in the spec.
    Returns a dictionary mapping node IDs to their PageRank. Each node
    should start with an initial PageRank value of 1/N, where N is the
    number of nodes in the graph.

    >>> g = graph.DirectedGraph()
    >>> g.add_node(0, airport_name='DTW')
    >>> g.add_node(1, airport_name='AMS', country='The Netherlands')
    >>> g.add_node(2, airport_name='ORD', city='Chicago')
    >>> g.add_edge(0, 1, flight_time_in_hours=8)
    >>> g.add_edge(0, 2, flight_time_in_hours=1)
    >>> g.add_edge(1, 0, airline_name='KLM')
    >>> abs(pagerank(g, 1)[0] - 0.427777) < 0.001
    True
    >>> abs(pagerank(g, 1)[1] - 0.286111) < 0.001
    True
    >>> abs(pagerank(g, 1)[2] - 0.286111) < 0.001
    True
    >>> abs(pagerank(g)[0] - 0.393617) < 0.001
    True
    >>> abs(pagerank(g)[1] - 0.303191) < 0.001
    True
    >>> abs(pagerank(g)[2] - 0.303191) < 0.001
    True
    """
    N = len(digraph)
    d = damping_factor
    nodes = digraph.nodes()
    sinkMask = sinkMaskGenerator(digraph)

    PR = np.zeros(N) + (1 / N)
    outDegreeBL = np.zeros(N, dtype=int)

    for j, n in enumerate(nodes):
        outDegreeBL[j] = digraph.out_degree(n.identifier())

    for _ in tqdm(range(num_iterations)):
        oldPR = np.array(PR)
        for i, u in enumerate(nodes):
            BLMask = backLinksMaskGenerator(digraph, u)

            PR[i] = ((1 - d) / N) + d * (np.sum(oldPR[BLMask] / outDegreeBL[BLMask]) + np.sum(oldPR[sinkMask] / N))
    return PR


def backLinksMaskGenerator(graph, u):
    mask = np.zeros(len(graph), dtype=int)
    nodes = graph.nodes()
    for e in graph.edges():
        n1, n2 = e.nodes()
        if n2 == u:
            mask[nodes.index(n1)] = 1
    return np.array(mask, dtype=bool)


def sinkMaskGenerator(graph):
    mask = np.zeros(len(graph), dtype=int)
    nodes = graph.nodes()
    for i, n in enumerate(nodes):
        if graph.out_degree(n.identifier()) == 0:
            mask[i] = 1
    return np.array(mask, dtype=bool)


def print_ranks(ranks, max_nodes=20):
    """Print out the top PageRanks in the given dictionary.

    Prints out the node IDs and rank values in the given dictionary,
    for the max_nodes highest ranked nodes, as well as the sum of all
    rank values. If max_nodes is not in [0, len(ranks)], prints out
    all nodes' rank values.
    """
    if max_nodes not in range(len(ranks)):
        max_nodes = len(ranks)
    # sort ids highest to lowest primarily by rank value, secondarily
    # by id itself
    sorted_ids = sorted(ranks.keys(),
                        key=lambda node: (round(ranks[node], 5), node),
                        reverse=True)
    for node_id in sorted_ids[:max_nodes]:
        print(f'{node_id}: {ranks[node_id]:.5f}')
    if max_nodes < len(ranks):
        print('...')
    # compute sum using sorted ids to bypass randomness in dict
    # implementation
    print(f'Sum: {(sum(ranks[n] for n in sorted_ids)):.5f}')


def pagerank_from_csv(node_file, edge_file, num_iterations):
    """Compute the PageRanks of the graph in the given files.

    Reads a digraph from CSV node/edge files in the format enumerated
    in the spec. Runs the PageRank algorithm for num_iterations on the
    resulting graph. Prints out the node ID and its PageRank for the
    20 highest-ranked nodes, or all nodes if the graph has fewer than
    20 nodes, to standard out. Also prints out the sum of all the
    PageRank values, which should approximate to 1.
    """
    rgraph = graph.read_graph_from_csv(node_file, edge_file, True)
    ranks = pagerank(rgraph, num_iterations)
    print_ranks(ranks)


def usage():
    """Print a usage string and exit."""
    print('Usage: python3 pagerank.py <node_file> <edge_file> ' +
          '[<num_iterations>]')
    sys.exit(1)


def main(*args):
    """Command-line interface for this module."""
    num_iterations = 40
    if len(args) < 2:
        usage()
    elif len(args) > 2:
        try:
            num_iterations = int(args[2])
        except ValueError:
            usage()
    pagerank_from_csv(args[0], args[1], num_iterations)

if __name__ == '__main__':
    # Reads a digraph from the node and edge files passed as
    # command-line arguments.
    # main(*sys.argv[1:])
    pagerank_from_csv('email-Eu-core.txt-nodes.csv', 'email-Eu-core.txt-edges.csv', 40)