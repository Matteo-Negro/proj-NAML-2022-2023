"""
graph.py.

ADT definitions for directed and undirected graphs.

Project UID bd3b06d8a60861e18088226c3a1f0595e4426dcf
"""

import csv
import doctest
import re

# add whatever imports you need here
import numpy as np


class GraphError(Exception):
    """This class is used for raising exceptions in the graph ADTs.

    >>> e = GraphError()
    >>> str(e)
    ''
    >>> f = GraphError('some message')
    >>> str(f)
    'some message'
    >>> repr(e)
    "GraphError('')"
    >>> f # interpreter does the same thing as print(repr(f))
    GraphError('some message')
    """

    def __init__(self, message=''):
        """Initialize this exception with the given message.

        The message defaults to an empty string.
        """
        super().__init__(message)
        self.message = message

    def __str__(self):
        """Return the message used to create this exception."""
        return f'{self.message}'

    def __repr__(self):
        """Return the canonical string representation of this graph."""
        return f'GraphError({repr(str(self.message))})'


class Node:
    r"""Represents a node in a graph.

    >>> n = Node('node1', weight=80, age=90)
    >>> n.identifier()
    'node1'
    >>> d = n.attributes()
    >>> d['age']
    90
    >>> d['weight']
    80
    >>> str(n)
    'Node [node1]\n    age : 90\n    weight : 80\n'
    """

    def __init__(self, identifier, **attributes):
        """Initialize this node with the given ID.

        The keyword arguments are optional node attributes.
        """
        self._Id = identifier
        self._attributes = {}
        for att in attributes:
            self._attributes[att] = attributes[att]

    def identifier(self):
        """Return the identifier of this node."""
        return self._Id

    def attributes(self):
        """Return a copy of this node's attribute dictionary."""
        return dict(self._attributes)

    def __str__(self):
        """Return a string representation of this node.

        Produces a representation of this node and its attributes in
        sorted, increasing, lexicographic order.
        """
        keys = list(self._attributes.keys())
        keys = sorted(keys, reverse=False)

        tmp = f'Node [{self._Id}]\n'
        for k in keys:
            tmp += '    '
            tmp += f'{k} : {self._attributes[k]}\n'

        return tmp


class Edge:
    r"""Represents a directed edge in a graph.

    >>> n1, n2 = Node('node1'), Node('node2')
    >>> e = Edge(n1, n2, size=3, cost=5)
    >>> d = e.attributes()
    >>> d['cost']
    5
    >>> d['size']
    3
    >>> e.nodes() == (n1, n2)
    True
    >>> str(e)
    'Edge from node [node1] to node [node2]\n    cost : 5\n    size : 3\n'
    """

    def __init__(self, node1, node2, **attributes):
        """Initialize this edge with the Nodes node1 and node2.

        The keyword arguments are optional edge attributes.
        """
        self._node1 = node1
        self._node2 = node2
        self._attributes = {}
        for att in attributes:
            self._attributes[att] = attributes[att]

    def attributes(self):
        """Return a copy of this edge's attribute dictionary."""
        return dict(self._attributes)

    def nodes(self):
        """Return a tuple of the Nodes corresponding to this edge.

        The nodes are in the same order as passed to the constructor.
        """
        return tuple([self._node1, self._node2])

    def __str__(self):
        """Return a string representation of this edge.

        Produces a representation of this edge and its attributes in
        sorted, increasing, lexicographic order.
        """
        keys = list(self._attributes.keys())
        keys = sorted(keys, reverse=False)

        tmp = f'Edge from node [{self._node1.identifier()}] to node [{self._node2.identifier()}]\n'
        for k in keys:
            tmp += '    '
            tmp += f'{k} : {self._attributes[k]}\n'

        return tmp


class BaseGraph:
    r"""A graph where the nodes and edges have optional attributes.

    This class should not be instantiated directly by a user.

    >>> g = BaseGraph()
    >>> len(g)
    0
    >>> g.add_node(1, a=1, b=2)
    >>> g.add_node(3, f=6, e=5)
    >>> g.add_node(2, c=3)
    >>> g.add_edge(1, 2, g=7)
    >>> g.add_edge(3, 2, h=8)
    >>> len(g)
    3
    >>> str(g.node(2))
    'Node [2]\n    c : 3\n'
    >>> g.node(4)
    Traceback (most recent call last):
        ...
    GraphError: ...
    >>> str(g.edge(1, 2))
    'Edge from node [1] to node [2]\n    g : 7\n'
    >>> g.edge(1, 3)
    Traceback (most recent call last):
        ...
    GraphError: ...
    >>> len(g.nodes())
    3
    >>> g.nodes()[0].identifier()
    1
    >>> len(g.edges())
    2
    >>> str(g.edges()[1])
    'Edge from node [3] to node [2]\n    h : 8\n'
    >>> 1 in g, 4 in g
    (True, False)
    >>> (1, 2) in g, (2, 3) in g
    (True, False)
    >>> g[1].identifier()
    1
    >>> g[(1, 2)].nodes()[0].identifier()
    1
    >>> print(g)
    BaseGraph:
    Node [1]
        a : 1
        b : 2
    Node [2]
        c : 3
    Node [3]
        e : 5
        f : 6
    Edge from node [1] to node [2]
        g : 7
    Edge from node [3] to node [2]
        h : 8
    <BLANKLINE>
    """

    def __init__(self):
        """Initialize this graph object."""
        self._NumNodes = 0
        self._nodes = {}
        self._edges = {}
        self._bl = {}

    def __len__(self):
        """Return the number of nodes in the graph."""
        return self._NumNodes

    def add_node(self, node_id, **attributes):
        """Add a node to this graph.

        Requires that node_id, the unique identifier for the node, is
        hashable and comparable to all identifiers for nodes currently
        in the graph. The keyword arguments are optional node
        attributes. Raises a GraphError if a node already exists with
        the given ID.
        """
        if node_id in self._nodes:
            raise GraphError('Node already exists with the given ID.')

        n = Node(node_id, **attributes)
        self._nodes[node_id] = n
        self._NumNodes += 1
        self._bl[n] = []

    def node(self, node_id):
        """Return the Node object for the node whose ID is node_id.

        Raises a GraphError if the node ID is not in the graph.
        """
        if node_id not in self._nodes:
            raise GraphError('Node ID is not in the graph.')

        return self._nodes[node_id]

    def nodes(self):
        """Return a list of all the Nodes in this graph.

        The nodes are sorted by increasing node ID.
        """
        keys = sorted(self._nodes.keys(), reverse=False)
        tmp = []
        for k in keys:
            tmp.append(self._nodes[k])

        return tmp

    def add_edge(self, node1_id, node2_id, **attributes):
        """Add an edge between the nodes with the given IDs.

        The keyword arguments are optional edge attributes. Raises a
        GraphError if either node is not found, or if the graph
        already contains an edge between the two nodes.
        """
        if node1_id not in self._nodes or node2_id not in self._nodes:
            raise GraphError('One of the node (or both) is (are) not in the graph.')

        e = Edge(self._nodes[node1_id], self._nodes[node2_id], **attributes)

        if e.nodes() in self._edges:
            raise GraphError('Graph already contains an edge between the two nodes.')
        self._edges[f'({node1_id}, {node2_id})'] = e

        self._bl[self._nodes[node2_id]].append(self._nodes[node1_id])

    def edge(self, node1_id, node2_id):
        """Return the Edge object for the edge between the given nodes.

        Raises a GraphError if the edge is not in the graph.
        """
        if f'({node1_id}, {node2_id})' not in self._edges:
            raise GraphError('Edge is not in the graph.')

        return self._edges[f'({node1_id}, {node2_id})']

    def edges(self):
        """Return a list of all the edges in this graph.

        The edges are sorted in increasing, lexicographic order of the
        IDs of the two nodes in each edge.
        """
        keys = sorted(self._edges.keys(), reverse=False)
        tmp = []
        for k in keys:
            tmp.append(self._edges[k])

        return tmp

    def get_bl(self):
        """Methods to evaluate BLs and return them."""
        result = {}
        nodes = self.nodes()
        idxs = {}

        for i, n in enumerate(nodes):
            idxs[n] = i

        for n in nodes:
            mask = np.zeros(len(self), dtype=bool)
            for k in self._bl[n]:
                mask[idxs[k]] = True

            result[n] = mask

        return result

    def __getitem__(self, key):
        """Return the Node or Edge corresponding to the given key.

        If key is a node ID, returns the Node object whose ID is key.
        If key is a pair of node IDs, returns the Edge object
        corresponding to the edge between the two nodes. Raises a
        GraphError if the node IDs or edge are not in the graph.
        """
        regex = re.compile('\(\w+, \w+\)|\(\w+, \'\w+\'\)|\(\'\w+\', \w+\)|\(\'\w+\', \'\w+\'\)')
        edge = regex.match(str(key))

        if edge:
            tmp = f'({key[0]}, {key[1]})'
            if tmp not in self._edges:
                raise GraphError('Edge is not in the graph.')
            return self._edges[tmp]

        if key not in self._nodes:
            raise GraphError('Node is not in the graph.')
        return self._nodes[key]

    def __contains__(self, item):
        """Return whether the given node or edge is in the graph.

        If item is a node ID, returns True if there is a node in this
        graph with ID item. If item is a pair of node IDs, returns
        True if there is an edge corresponding to the two nodes.
        Otherwise, returns False.
        """
        regex = re.compile('\(\w+, \w+\)|\(\w+, \'\w+\'\)|\(\'\w+\', \w+\)|\(\'\w+\', \'\w+\'\)')
        edge = regex.match(str(item))
        if edge:
            tmp = f'({item[0]}, {item[1]})'
            if tmp in self._edges:
                return True
        else:
            if item in self._nodes:
                return True
        return False

    def __str__(self):
        """Return a string representation of the graph.

        The string representation contains the nodes in sorted,
        increasing order, followed by the edges in order.
        """
        result = f'{type(self).__name__}:\n'
        for node in self.nodes():
            result += str(node)
        for edge in self.edges():
            result += str(edge)
        return result


class UndirectedGraph(BaseGraph):
    """An undirected graph where nodes/edges have optional attributes.

    >>> g = UndirectedGraph()
    >>> g.add_node(1, a=1)
    >>> g.add_node(2, b=2)
    >>> g.add_edge(1, 2, c=3)
    >>> len(g)
    2
    >>> g.degree(1)
    1
    >>> g.degree(2)
    1
    >>> g.edge(1, 2).nodes() == (g.node(1), g.node(2))
    True
    >>> g.edge(2, 1).nodes() == (g.node(2), g.node(1))
    True
    >>> 1 in g, 4 in g
    (True, False)
    >>> (1, 2) in g, (2, 1) in g
    (True, True)
    >>> (2, 3) in g
    False
    >>> g[1].identifier()
    1
    >>> g[(1, 2)].nodes()[0].identifier()
    1
    >>> g.add_edge(1, 1, d=4)
    Traceback (most recent call last):
        ...
    GraphError: ...
    >>> print(g)
    UndirectedGraph:
    Node [1]
        a : 1
    Node [2]
        b : 2
    Edge from node [1] to node [2]
        c : 3
    Edge from node [2] to node [1]
        c : 3
    <BLANKLINE>
    """

    def degree(self, node_id):
        """Return the degree of the node with the given ID.

        Raises a GraphError if the node ID is not found.
        """
        if node_id not in self._nodes:
            raise GraphError('Node ID is not found.')

        regex = re.compile(
            f'\({node_id}, \w+\)|\(\'{node_id}\', \w+\)|\({node_id}, \'\w+\'\)|\(\'{node_id}\', \'\w+\'\)')
        edges = list(self._edges.keys())

        degree = 0
        for e in edges:
            if regex.match(str(e)):
                degree += 1

        return degree

    def add_edge(self, node1_id, node2_id, **attributes):
        """Add two edge between the nodes with the given IDs (Both directions).

        The keyword arguments are optional edge attributes. Raises a
        GraphError if either node is not found, or if the graph
        already contains an edge between the two nodes.
        """
        if node1_id == node2_id:
            raise GraphError("Self-loop")
        super().add_edge(node1_id, node2_id, **attributes)
        super().add_edge(node2_id, node1_id, **attributes)


class DirectedGraph(BaseGraph):
    """A directed graph where nodes/edges have optional attributes.

    >>> g = DirectedGraph()
    >>> g.add_node(1, a=1)
    >>> g.add_node(2, b=2)
    >>> g.add_edge(1, 2, c=3)
    >>> len(g)
    2
    >>> g.in_degree(1), g.out_degree(1)
    (0, 1)
    >>> g.in_degree(2), g.out_degree(2)
    (1, 0)
    >>> g.edge(1, 2).nodes() == (g.node(1), g.node(2))
    True
    >>> g.edge(2, 1)
    Traceback (most recent call last):
        ...
    GraphError: ...
    >>> 1 in g, 4 in g
    (True, False)
    >>> (1, 2) in g, (2, 1) in g
    (True, False)
    >>> g[1].identifier()
    1
    >>> g[(1, 2)].nodes()[0].identifier()
    1
    >>> g.add_edge(1, 1, d=4)
    >>> g.in_degree(1), g.out_degree(1)
    (1, 2)
    >>> g.in_degree(-1)
    Traceback (most recent call last):
        ...
    GraphError: ...
    >>> g.out_degree(-1)
    Traceback (most recent call last):
        ...
    GraphError: ...
    >>> print(g)
    DirectedGraph:
    Node [1]
        a : 1
    Node [2]
        b : 2
    Edge from node [1] to node [1]
        d : 4
    Edge from node [1] to node [2]
        c : 3
    <BLANKLINE>
    """

    def __init__(self):
        """Initialize this digraph object."""
        super().__init__()
        self._OutDegreeDict = {}

    def in_degree(self, node_id):
        """Return the in-degree of the node with the given ID.

        Raises a GraphError if the node is not found.
        """
        if node_id not in self._nodes:
            raise GraphError('Node ID is not found.')

        regex = re.compile(
            f'\(\w+, {node_id}\)|\(\w+, \'{node_id}\'\)|\(\'\w+\', {node_id}\)|\(\'\w+\', \'{node_id}\'\)')
        edges = list(self._edges.keys())

        degree = 0
        for e in edges:
            if regex.match(str(e)):
                degree += 1

        return degree

    def out_degree(self, node_id):
        """Return the out-degree of the node with the given ID.

        Raises a GraphError if the node is not found.
        """
        if node_id not in self._nodes:
            raise GraphError('Node ID is not found.')

        regex = re.compile(
            f'\({node_id}, \w+\)|\(\'{node_id}\', \w+\)|\({node_id}, \'\w+\'\)|\(\'{node_id}\', \'\w+\'\)')
        edges = list(self._edges.keys())

        degree = 0
        for e in edges:
            if regex.match(str(e)):
                degree += 1

        return degree

    def out_degree_vector(self):
        """Return the out-degree of all the nodes."""
        nodes = self.nodes()
        degree = np.zeros(len(nodes), dtype=int)

        for i, n in enumerate(nodes):
            degree[i] = self._OutDegreeDict[n]

        return degree

    def add_node(self, node_id, **attributes):
        """Add a node to this graph.

        Requires that node_id, the unique identifier for the node, is
        hashable and comparable to all identifiers for nodes currently
        in the graph. The keyword arguments are optional node
        attributes. Raises a GraphError if a node already exists with
        the given ID.
        """
        super().add_node(node_id, **attributes)
        self._OutDegreeDict[self._nodes[node_id]] = 0

    def add_edge(self, node1_id, node2_id, **attributes):
        """Add the edge between the nodes with the given IDs.

        The keyword arguments are optional edge attributes. Raises a
        GraphError if either node is not found, or if the graph
        already contains an edge between the two nodes.
        """
        super().add_edge(node1_id, node2_id, **attributes)
        self._OutDegreeDict[self._nodes[node1_id]] += 1


def read_graph_from_csv(node_file, edge_file, directed=False):
    """Read a graph from CSV node and edge files.

    Refer to the project specification for the file formats.
    """
    result = DirectedGraph() if directed else UndirectedGraph()
    for i, filename in enumerate((node_file, edge_file)):
        attr_start = i + 1
        with open(filename, 'r', encoding="utf8") as csv_data:
            reader = csv.reader(csv_data)
            header = next(reader)
            attr_names = header[attr_start:]

            for line in reader:
                identifier, attr_values = (line[:attr_start],
                                           line[attr_start:])
                attributes = {attr_names[i]: attr_values[i]
                              for i in range(len(attr_names))}
                if i == 0:
                    result.add_node(*identifier, **attributes)
                else:
                    result.add_edge(*identifier, **attributes)
    return result


def _test():
    """Run this module's doctests."""
    doctest.testmod(optionflags=doctest.IGNORE_EXCEPTION_DETAIL)


if __name__ == '__main__':
    # Run the doctests
    _test()
