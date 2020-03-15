import argparse

import numpy as np
import graphviz

class Node:
    """
    Node representation of network
    """
    def __init__(self, node_id: int):
        """
        Parameters:
            node_id (int): Unique label
        """
        self.id = node_id
        self.label = str(node_id)
        self.neighbours = {}
    
    def add_neighbour(self, neighbour: "Node", edge_cost: float):
        """
        Add neighbouring node and corresponding edge cost to node information.

        Parameters:
            neighbour (Node): neighbouring node
            edge_cost (float): cost of moving from node to neighbour
        """
        self.neighbours[neighbour] = edge_cost

    def __eq__(self, other: "Node"):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __str__(self):
        neighbours_string = str([(node.id, edge_cost) for node, edge_cost in self.neighbours.items()])
        return ",".join([self.label, neighbours_string])


class Network:
    def __init__(
        self,
        num_nodes: int, 
        num_edges_per_node: int
        ):

        self.nodes = []
        self.num_nodes = num_nodes
        self.initialise_network(num_nodes, num_edges_per_node)

    def initialise_network(self, num_nodes, num_edges_per_node):
        """
        Create network as a list of nodes, with edge information stored
        in each node.

        Parameters:
            num_nodes (int): Number of nodes in the network.
            num_edges_per_node (int): Number of neighbours added to each node.
        """
        # Initialise nodes
        for node_id in range(num_nodes):
            self.nodes.append(Node(node_id))

        # Initialise edges
        for i, node in enumerate(self.nodes):
            # Generate edges without replacement from list of other nodes
            neighbour_indices = np.random.choice(
                [node_id for node_id in range(self.num_nodes) if node_id != node.id], 
                num_edges_per_node,
                replace=False
                )
            for neighbour_idx in neighbour_indices:
                edge_cost = np.random.rand()
                node.add_neighbour(self.nodes[neighbour_idx], edge_cost)
                self.nodes[neighbour_idx].add_neighbour(node, edge_cost)

    def __str__(self):
        return str([str(node) for node in self.nodes])

    def visualise_network(self, output_location: str, output_format: str = "svg"):
        """
        Visualise network using graphviz by saving to file.

        Parameters:
            output_location (str): File path to write graph visualisation output to.
            output_format (str): File format of visualisation output (E.g. "png", "svg")
        """
        graph = graphviz.Graph(format=output_format)

        # Set to keep track of nodes that have been seen so avoid duplicating edges.
        seen_nodes = set()

        for node in self.nodes:
            graph.node(node.label)
            seen_nodes.add(node)

            for neighbour, edge_cost in node.neighbours.items():
                # Generate edge only if we have not seen it yet.
                if neighbour not in seen_nodes:
                    graph.edge(node.label, neighbour.label, label = "{:.2g}".format(edge_cost))

        graph.render(output_location, view=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate network.')
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--num-edges-per-node", type=int, default=1)
    parser.add_argument("--output-location", type=str, default ="graphs/test")

    args = parser.parse_args()

    network = Network(args.num_nodes, args.num_edges_per_node)
    network.visualise_network(args.output_location)

        

        
    


    