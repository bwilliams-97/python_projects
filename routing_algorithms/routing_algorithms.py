import argparse
from copy import deepcopy
from typing import Dict, Tuple, Union

from base_routing_algorithm import BaseRoutingAlgorithm, ShortestPath
from network_generator import Network, Node

class BellmanFordRouter(BaseRoutingAlgorithm):
    def __init__(self, network: Network):
        super(BellmanFordRouter, self).__init__(network)

    def find_shortest_paths(self, target_node_idx) -> None:
        """
        Obtain shortest paths from every node to target node.
        Parameters:
            target_node_idx (int): Index of target node to which paths are found.
        """

        self.initialise(target_node_idx)

        for i in range(len(self.network.nodes)):
            self.iterate()

        for node, path_detail in self.path_costs.items():
            print(node.label, path_detail.cost, str(path_detail.path_node))

    def initialise(self, target_node_idx: int) -> None:
        """
        Initialise path costs. Path weights are initialised as inf if there is no direct link from node to target node,
        else as edge cost between node and target node.

        Next node on path is initialised to -1 if no direct link, else target node.

        Parameters:
            target_node_idx (int): Index of target node to which paths are found.
        """
        self.target_node: Node = self.network.nodes[target_node_idx]

        # Maps a node to a tuple of (Next node on path to target, cost of this path)
        self.path_costs: Dict[Node, ShortestPath] = {}

        for node in self.network.nodes:
            # Trivial case
            if node == self.target_node:
                self.path_costs[node] = ShortestPath(0, node)
                continue
            
            self.path_costs[node] = ShortestPath(node.neighbours[self.target_node], self.target_node) if self.target_node in node.neighbours else ShortestPath(float('inf'), -1)
    
    def iterate(self) -> None:
        """
        Each iteration, every node is updated with the path cost for its neighbouring nodes.
        If the cost of going to the target via a neighbour is less than the current path cost,
        it updates its current path with next node as the neighbour.
        """
        # Deepcopy so that we can update self.path_costs simultaneously for all nodes
        current_path_costs = deepcopy(self.path_costs)

        for node in self.network.nodes:
            for neighbour, edge_cost in node.neighbours.items():
                # If path via neighbour is better than current path, update.
                if edge_cost + current_path_costs[neighbour].cost < current_path_costs[node].cost:
                    self.path_costs[node] = ShortestPath(edge_cost + current_path_costs[neighbour].cost, neighbour)

    def find_path_to_target(self, start_node_idx) -> str:
        """
        Find shortest path by following next nodes along path in self.path_costs.
        """
        start_node = self.network.nodes[start_node_idx]

        complete_path_cost = self.path_costs[start_node].cost
        if complete_path_cost == float('inf'):
            # If graph contains disjoint subgraphs
            return "No connection between start node and target node"

        shortest_path = []
        shortest_path.append(start_node.label)
       
        next_node = self.path_costs[start_node].path_node
        while next_node != self.target_node:
            shortest_path.append(next_node.label)
            next_node = self.path_costs[next_node].path_node
        
        shortest_path.append(self.target_node.label)

        return "->".join(shortest_path)

class DijkstraRouter(BaseRoutingAlgorithm):
    def __init__(self, network: Network):
        super(DijkstraRouter, self).__init__(network)

    def find_shortest_paths(self, start_node_idx: int) -> None:
        """
        Find shortest paths to all nodes from start node.
        We keep track of nodes we know the path for and those we don't.
        We iterate until we know the path for all nodes.

        Parameters:
            start_node_idx (int): Index of start node from which paths are found.
        """
        
        # Maps a node to a tuple of (Previous node on path to target, cost of this path)
        self.path_costs: Dict[Node, ShortestPath]= {}

        self.start_node: Node = self.network.nodes[start_node_idx]

        self.disjoint: bool = False
        
        self.nodes_with_known_path: set = set()
        self.nodes_without_known_path: set = set(self.network.nodes)

        self.initialise()
        while(len(self.nodes_with_known_path) < len(self.network.nodes)):
            if not self.iterate():
                self.disjoint = True
                break
        
        for node, path_detail in self.path_costs.items():
            print(node.label, path_detail.cost, str(path_detail.path_node))

    def initialise(self) -> None:
        """
        Initialise path costs. Path weights are initialised as inf if there is no direct link from start node to node,
        else as edge cost between start node and node.

        Previous node on path is initialised to -1 if no direct link, else target node.

        Parameters:
            target_node_idx (int): Index of target node to which paths are found.
        """
        # Add start node to known set
        self.nodes_with_known_path.add(self.start_node)
        self.nodes_without_known_path.remove(self.start_node)   

        # Trivial case
        self.path_costs[self.start_node] = ShortestPath(0, self.start_node)

        for node in self.nodes_without_known_path:
            self.path_costs[node] = ShortestPath(node.neighbours[self.start_node], self.start_node) if self.start_node in node.neighbours else ShortestPath(float("inf"), -1)

    def iterate(self) -> None:
        """
        Each iteration we add a new node (node_to_add) to the set of nodes with known paths.
        This is the node with the minimum path cost of those nodes not in the set.
        Then update all costs and previous nodes for those not in the set.
            If current path better than going via node_to_add, do nothing.
            Else update path and path cost as going via node_to_add
        """
        node_to_add = None
        min_cost = float("inf")
        # Find node in set without known paths with shortest current path.
        for node in self.nodes_without_known_path:
            current_node_cost = self.path_costs[node].cost
            if current_node_cost < min_cost:
                node_to_add = node
                min_cost = current_node_cost

        if type(node_to_add) == type(None):
            print("Warning: disjoint graphs")
            return False

        self.nodes_with_known_path.add(node_to_add)
        self.nodes_without_known_path.remove(node_to_add)

        # Iterate through remaining nodes and see if they can update their path by
        # going via node_to_add
        for node in self.nodes_without_known_path:
            if node in node_to_add.neighbours:
                alternative_path_cost = self.path_costs[node_to_add].cost + node_to_add.neighbours[node]
                if alternative_path_cost < self.path_costs[node].cost:
                    self.path_costs[node] = ShortestPath(alternative_path_cost, node_to_add)
        
        return True

    def find_path_to_target(self, target_node_idx: int) -> str:
        """
        Find shortest path back to start node by following previous nodes along path in self.path_costs.
        """
        if self.disjoint:
            return "Graph contains disjoint subgraphs."

        self.target_node: Node = self.network.nodes[target_node_idx]

        shortest_path = []
        shortest_path.insert(0, str(self.target_node))

        previous_node = self.path_costs[self.target_node].path_node
        while previous_node != self.start_node:
            shortest_path.insert(0, str(previous_node))
            previous_node = self.path_costs[previous_node].path_node

        shortest_path.insert(0, str(self.start_node))            

        return "->".join(shortest_path), self.path_costs[self.target_node].cost

def main():
    parser = argparse.ArgumentParser(description='Generate network.')
    parser.add_argument("--num-nodes", type=int, default=10)
    parser.add_argument("--num-edges-per-node", type=int, default=1)
    parser.add_argument("--vis-output-location", type=str, default ="graphs/test")
    parser.add_argument("--routing-algorithm", type=str, default="dijkstra")
    parser.add_argument("--start-node-idx", type=int, default=0)
    parser.add_argument("--target-node-idx", type=int, default=5)

    args = parser.parse_args()

    network = Network(args.num_nodes, args.num_edges_per_node)

    if args.routing_algorithm == "dijkstra":
        router = DijkstraRouter(network)
        router.find_shortest_paths(args.start_node_idx)
        print("Shortest path: ", router.find_path_to_target(args.target_node_idx))

    elif args.routing_algorithm == "bellman-ford":
        router = BellmanFordRouter(network)
        router.find_shortest_paths(args.target_node_idx)
        print("Shortest path: ", router.find_path_to_target(args.start_node_idx))

    network.visualise_network(args.vis_output_location)

if __name__ == "__main__":
    main()
