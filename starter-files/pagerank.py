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
import time


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
    PR = np.zeros(N) + (1 / N)

    print('Getting Nodes...')
    t0 = time.time_ns()
    nodes = digraph.nodes()
    delta = time.time_ns() - t0
    print(f'Getting Nodes... DONE [{(delta * 1e-9): 0.5f}]')

    print('Evaluating BackLinks...')
    t0 = time.time_ns()
    # BLMask = backLinksMaskGenerator(digraph)
    BLMask = digraph.getBL()
    delta = time.time_ns() - t0
    print(f'Evaluating BackLinks... DONE [{(delta * 1e-9): 0.5f}]')

    print('Evaluating OutDegree...')
    # outDegreeVec = np.zeros(N, dtype=int)
    # for j, n in enumerate(nodes):
    #     outDegreeVec[j] = digraph.out_degree(n.identifier())
    t0 = time.time_ns()
    outDegreeVec = digraph.out_degree_vector_2()
    delta = time.time_ns() - t0
    print(f'Evaluating OutDegree... DONE [{(delta * 1e-9): 0.5f}]')

    print('Evaluating SinkMask...')
    t0 = time.time_ns()
    sinkMask = outDegreeVec == 0
    delta = time.time_ns() - t0
    print(f'Evaluating SinkMask... DONE [{(delta * 1e-9): 0.5f}]')

    print('Computing the Alg...')
    t0 = time.time_ns()
    for _ in range(num_iterations):
        oldPR = np.array(PR)
        sinkFactor = np.sum(oldPR[sinkMask] / N)
        for i, u in enumerate(nodes):
            PR[i] = ((1 - d) / N) + d * (np.sum(oldPR[BLMask[u]] / outDegreeVec[BLMask[u]]) + sinkFactor)
    delta = time.time_ns() - t0
    print(f'Computing the Alg... DONE [{(delta * 1e-9): 0.5f}]')

    print('Preparing the printing...')
    t0 = time.time_ns()
    result = dict()
    for j, n in enumerate(nodes):
        result[n.identifier()] = PR[j]
    delta = time.time_ns() - t0
    print(f'Preparing the printing... DONE [{(delta * 1e-9): 0.5f}]')

    return result


def backLinksMaskGenerator(graph):
    result = dict()
    nodes = graph.nodes()
    for n in nodes:
        result[n] = np.zeros(len(graph), dtype=bool)

    for e in graph.edges():
        n1, n2 = e.nodes()
        result[n2][nodes.index(n1)] = True

    return result


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
    readTime = time.time_ns()
    rgraph = graph.read_graph_from_csv(node_file, edge_file, True)
    execTime = time.time_ns()
    ranks = pagerank(rgraph, num_iterations)
    printTime = time.time_ns()
    print('-----------------------------------------')
    print_ranks(ranks)
    endTime = time.time_ns()

    print(' ---------------------------------------')
    print(f'|\tREAD TIME: \t\t{execTime - readTime}\t\t\t|')
    print(f'|\tEXEC TIME: \t\t{printTime - execTime}\t\t\t|')
    print(f'|\tPRINT TIME: \t{endTime - printTime}\t\t\t\t|')
    print('|\tThese times are collected in ns.\t|')
    print(' ---------------------------------------')

    print(' ---------------------------------------')
    print(f'|\tREAD TIME: \t\t{((execTime - readTime) * 1e-9): .5f}\t\t\t|')
    print(f'|\tEXEC TIME: \t\t{((printTime - execTime) * 1e-9): .5f}\t\t\t|')
    print(f'|\tPRINT TIME: \t{((endTime - printTime) * 1e-9): .5f}\t\t\t|')
    print('|\tThese times are collected in s.\t\t|')
    print(' ---------------------------------------')


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
    pagerank_from_csv('twitter_combined.txt-nodes.csv', 'twitter_combined.txt-edges.csv', 40)
