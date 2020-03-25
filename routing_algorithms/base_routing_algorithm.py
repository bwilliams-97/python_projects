from typing import Union
from abc import ABC, abstractmethod

from network_generator import Network, Node

class BaseRoutingAlgorithm(ABC):
    def __init__(self, network: Network):
        self.network: Network = network

    @abstractmethod
    def find_shortest_paths(self, node_idx: int) -> None:
        """
        Find shortest paths from/to a specified node.
        Should call initialise and iterate.
        """
        pass
    
    @abstractmethod
    def initialise(self, node_idx: int) -> None:
        pass

    @abstractmethod
    def iterate(self) -> None:
        pass

    @abstractmethod
    def find_path_to_target(self, node_idx) -> str:
        pass

class ShortestPath():
    def __init__(self, path_cost: float, path_node: Union[Node, int]):
        """
        Keep track of path details.
        """
        self.cost: float = path_cost # Cost of specified path
        self.path_node: Node = path_node # Next node along path