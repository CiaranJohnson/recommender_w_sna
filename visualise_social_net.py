import networkx as nx

G=nx.Graph()
G.add_edges_from([(1,2),(1,3),(1,4),(3,4)])
G.nodes(data=True)
nx.write_graphml(G,'so.graphml')