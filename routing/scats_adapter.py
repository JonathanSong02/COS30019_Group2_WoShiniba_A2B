"""Convert *verified* SCATS roads and predicted travel times to A2A Graph.

Important: site coordinates alone DO NOT tell us which roads exist.
Darren's road-network module must supply real directed connections and
travel-time weights in seconds. Do not call Graph.distance() to find km:
A2A uses Euclidean XY; SCATS longitude/latitude requires a geographic
distance calculation in the travel-time preprocessing module.
"""

from graph import Graph


def build_weighted_scats_graph(site_coords, road_edges_seconds):
    """Arguments:
        site_coords: {site_id: (latitude, longitude)}
        road_edges_seconds: iterable of (from_id, to_id, seconds)

    Uses A2A Graph adjacency API; DOES NOT infer road connectivity.
    """
    graph = Graph()
    for site_id, (lat, lon) in site_coords.items():
        graph.add_node(int(site_id), float(lat), float(lon))
    for source, destination, seconds in road_edges_seconds:
        if int(source) not in graph.coords or int(destination) not in graph.coords:
            raise ValueError(f"Road connection has unknown site: {source}->{destination}")
        graph.add_edge(int(source), int(destination), seconds)
    graph.finalise()
    return graph
