from network_generator import Network

class BaseRoutingAlgorithm:
    def __init__(self, network: Network):
        self.network = network

    def print_shortest_paths():
        pass

class BellmanFordRouter(BaseRoutingAlgorithm):
    def __init__(self, network: Network):
        super(BellmanFordRouter, self).__init__(network)

class DijkstraRouter(BaseRoutingAlgorithm):
    def __init__(self, network: Network):
        super(DijkstraRouter, self).__init__(network)

    def find_shortest_paths(self, start_node_idx: int):
            
        self.path_costs = {}

        self.start_node = self.network.nodes[start_node_idx]

        self.disjoint = False
        
        self.nodes_with_known_path = set()
        self.nodes_without_known_path = set(self.network.nodes)

        self.initialise()
        while(len(self.nodes_with_known_path) < len(self.network.nodes)):
            if not self.iterate():
                self.disjoint = True
                break
        
        for node, path_detail in self.path_costs.items():
            print(node.label, path_detail[0], str(path_detail[1]))

    def find_path_to_target(self, target_node_idx: int):
        if self.disjoint:
            return "Graph is disjoint."
        self.target_node = self.network.nodes[target_node_idx]

        shortest_path = []
        shortest_path.insert(0, str(self.target_node))

        previous_node = self.path_costs[self.target_node][1]
        shortest_path.insert(0, str(previous_node))
        while previous_node != self.start_node:
            previous_node = self.path_costs[previous_node][1]
            shortest_path.insert(0, str(previous_node))

        return "->".join(shortest_path), self.path_costs[self.target_node][0]

    def initialise(self):
        """
        Need:
            target node
            dictionary mapping -> node: cost, previous node
        """
        # Add start node to known set
        self.nodes_with_known_path.add(self.start_node)
        self.nodes_without_known_path.remove(self.start_node)   

        # Trivial case
        self.path_costs[self.start_node] = (0, self.start_node)

        for node in self.nodes_without_known_path:
            cost, previous = (node.neighbours[self.start_node], self.start_node) if self.start_node in node.neighbours else (float("inf"), -1)
            self.path_costs[node] = (cost, previous) 

    def iterate(self):
        """
        For each iteration:
        1) Add new node to finalised_nodes with argmin cost i not in K
        2) Update costs and previous for all nodes not in K
            For each i not in K, if current path better then do nothing
            Otherwise, new path cost is current plus interim cost
        3) If K=N, done; o.w. repeat.
        """
        node_min = None
        min_cost = float("inf")
        for node in self.nodes_without_known_path:
            current_node_cost = self.path_costs[node][0]
            if current_node_cost < min_cost:
                node_min = node
                min_cost = current_node_cost

        if type(node_min) == type(None):
            print("Warning: disjoint graphs")
            return False

        self.nodes_with_known_path.add(node_min)
        self.nodes_without_known_path.remove(node_min)

        for node in self.nodes_without_known_path:
            if node in node_min.neighbours:
                alternative_path_cost = self.path_costs[node_min][0] + node_min.neighbours[node]
                if alternative_path_cost < self.path_costs[node][0]:
                    self.path_costs[node] = alternative_path_cost, node_min        
        
        return True

def main():
    network = Network(10, 1)

    router = DijkstraRouter(network)

    router.find_shortest_paths(0)

    print(router.find_path_to_target(5))

    network.visualise_network("graphs/test")

if __name__ == "__main__":
    main()
