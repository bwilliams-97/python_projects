import argparse

import numpy as np
import graphviz

class Node:
    def __init__(self, node_id: int):
        self.id = node_id
        self.neighbours = {}
    
    def add_neighbour(self, neighbour: "Node", edge_cost: float):
        self.neighbours[neighbour]= edge_cost

    def __eq__(self, other: "Node"):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __str__(self):
        neighbours_string = str([(node.id, edge_cost) for node, edge_cost in self.neighbours.items()])
        return ",".join([str(self.id), neighbours_string])


class Network:
    def __init__(
        self,
        num_nodes: int, 
        num_edges_per_node: int
        ):

        self.nodes = []
        self.initialise_network(num_nodes, num_edges_per_node)

    def initialise_network(self, num_nodes, num_edges_per_node):
        # Initialise nodes
        for node_id in range(num_nodes):
            self.nodes.append(Node(node_id))

        # Initialise edges
        for i, node in enumerate(self.nodes):
            edge_indices = np.random.randint(0, len(self.nodes), size=num_edges_per_node)
            for edge in edge_indices:
                edge_cost = np.random.rand()
                node.add_neighbour(self.nodes[edge], edge_cost)
                self.nodes[edge].add_neighbour(node, edge_cost)

    def __str__(self):
        return str([str(node) for node in self.nodes])

    def visualise(self):
        graph = graphviz.Graph(format="png")

        for node in self.nodes:
            graph.node(node.id)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate network.')
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--num-edges-per-node", type=int, default=1)

    args = parser.parse_args()

    network = Network(args.num_nodes, args.num_edges_per_node)
    print(network)

        

        
    


    