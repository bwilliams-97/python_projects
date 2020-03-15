from network_generator import Network

class BaseRoutingAlgorithm:
    def __init__(network: Network):
        self.network = network

    def print_shortest_paths():
        pass

class BellmanFordRouter(BaseRoutingAlgorithm):
    def __init__(self, network: Network):
        super(BaseRoutingAlgorithm, self).__init__(network)

class DijkstraRouter(BaseRoutingAlgorithm):
    def __init__(self, network: Network):
        super(BaseRoutingAlgorithm, self).__init__(network)