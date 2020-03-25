# Routing algorithms

This project implements well-known routing algorithms to find the shortest path between two nodes in a network.

### Network
[network_generator](network_generator.py) contains the graph and node classes to generate a network with random initialisation, given a specified number of nodes and number of edge per node (a given node may finish with more edges than this - this is a proxy used for network generation). These can be visualised using graphviz.

### Implementations
[routing_algorithms](routing_algorithms.py) contains implementations for [Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) and [Dijkstra's](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) algorithms. Each is implemented based on notes from the 3F4 Data Transmission course from Cambridge University Engineering Department (see course [details](http://teaching.eng.cam.ac.uk/content/engineering-tripos-part-iia-3f4-data-transmission-2017-18)). 