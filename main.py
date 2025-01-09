import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import math
import folium  # For real map visualization

# Real-world port data (latitude, longitude, and additional fees)
ports = {
    "Shanghai": {"lat": 31.2304, "lon": 121.4737, "port_fee": 500},
    "Rotterdam": {"lat": 51.9225, "lon": 4.47917, "port_fee": 600},
    "Los Angeles": {"lat": 33.7490, "lon": -118.2915, "port_fee": 700},
    "Singapore": {"lat": 1.3521, "lon": 103.8198, "port_fee": 400},
    "Dubai": {"lat": 25.276987, "lon": 55.296249, "port_fee": 550},
    "New York": {"lat": 40.7128, "lon": -74.0060, "port_fee": 800},
}

# Approximate distances between ports (in kilometers)
distances = [
    ("Shanghai", "Singapore", 3800),
    ("Shanghai", "Los Angeles", 10400),
    ("Rotterdam", "New York", 5800),
    ("Rotterdam", "Dubai", 5200),
    ("Los Angeles", "New York", 3900),
    ("Singapore", "Dubai", 5800),
    ("Dubai", "Rotterdam", 5200),
    ("New York", "Los Angeles", 3900),
]

# Sample shipments
shipments = [
    {
        "id": 1,
        "origin": "Shanghai",
        "destination": "Rotterdam",
        "weight": 100,
        "carrier": "Carrier X",
        "value": 10000,  # Shipment value for insurance
    },
    {
        "id": 2,
        "origin": "Los Angeles",
        "destination": "New York",
        "weight": 200,
        "carrier": "Carrier Y",
        "value": 20000,  # Shipment value for insurance
    },
]

# Carrier rates and additional charges
carrier_rates = {
    "Carrier X": {
        "rate_per_km": 0.5,
        "service_level": "Express",
        "fuel_surcharge": 0.1,
        "speed": 50,  # Average speed in km/h
    },
    "Carrier Y": {
        "rate_per_km": 0.3,
        "service_level": "Standard",
        "fuel_surcharge": 0.05,
        "speed": 40,  # Average speed in km/h
    },
}

# Additional fees
customs_fees = 200  # Flat fee for customs clearance
insurance_rate = 0.01  # 1% of shipment value
seasonal_surcharge = 0.05  # 5% during peak season


# Create a graph with real-world ports and distances
def create_graph_with_real_world_ports():
    G = nx.Graph()
    for port1, port2, distance in distances:
        G.add_edge(port1, port2, weight=distance)
    return G


# Heuristic function for A* (Euclidean distance based on latitude and longitude)
def heuristic(node1, node2):
    lat1, lon1 = ports[node1]["lat"], ports[node1]["lon"]
    lat2, lon2 = ports[node2]["lat"], ports[node2]["lon"]
    # Approximate distance using Haversine formula
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Function to optimize route using A* algorithm
def optimize_route_astar(graph, origin, destination):
    try:
        route = nx.astar_path(
            graph,
            source=origin,
            target=destination,
            heuristic=heuristic,
            weight="weight",
        )
        distance = nx.astar_path_length(
            graph,
            source=origin,
            target=destination,
            heuristic=heuristic,
            weight="weight",
        )
        return route, distance
    except nx.NetworkXNoPath:
        return None, None


# Function to calculate total freight cost
def calculate_total_freight_cost(shipment, distance, carrier_rate):
    base_cost = distance * carrier_rate
    fuel_surcharge = base_cost * carrier_rates[shipment["carrier"]]["fuel_surcharge"]
    port_fees = (
        ports[shipment["origin"]]["port_fee"]
        + ports[shipment["destination"]]["port_fee"]
    )
    insurance_cost = shipment["value"] * insurance_rate
    seasonal_cost = base_cost * seasonal_surcharge
    total_cost = (
        base_cost
        + fuel_surcharge
        + port_fees
        + customs_fees
        + insurance_cost
        + seasonal_cost
    )
    return total_cost


# Function to estimate delivery time
def estimate_delivery_time(distance, carrier_speed):
    # Assume 8 hours of operation per day
    hours_per_day = 8
    total_hours = distance / carrier_speed
    days = total_hours / hours_per_day
    return timedelta(days=days)


# Function to list all available ports
def list_available_ports():
    print("Available Ports:")
    for port, data in ports.items():
        print(f"- {port} (Port Fee: ${data['port_fee']})")


# Function to list shipping options by cost, distance, and time
def list_shipping_options(origin, destination, weight, value):
    graph = create_graph_with_real_world_ports()
    route, distance = optimize_route_astar(graph, origin, destination)
    if not route:
        print(f"No valid route found from {origin} to {destination}.")
        return

    print(f"\nShipping Options from {origin} to {destination}:")
    for carrier, details in carrier_rates.items():
        # Create a complete shipment dictionary
        shipment = {
            "carrier": carrier,
            "value": value,
            "origin": origin,
            "destination": destination,
        }
        total_cost = calculate_total_freight_cost(
            shipment, distance, details["rate_per_km"]
        )
        delivery_time = estimate_delivery_time(distance, details["speed"])
        print(
            f"Carrier: {carrier} ({details['service_level']}), "
            f"Cost: ${total_cost:.2f}, "
            f"Distance: {distance:.2f} km, "
            f"Estimated Delivery Time: {delivery_time.days} days"
        )


# Function to process shipments
def process_shipments(shipments, graph, carrier_rates):
    for shipment in shipments:
        # Optimize route using A*
        route, distance = optimize_route_astar(
            graph, shipment["origin"], shipment["destination"]
        )
        if route:
            shipment["route"] = route
            shipment["distance"] = distance
            print(
                f"Shipment ID {shipment['id']}: Optimized route from {shipment['origin']} to {shipment['destination']} is {route} ({distance:.2f} km)."
            )

            # Calculate total freight cost
            carrier = shipment["carrier"]
            rate = carrier_rates[carrier]["rate_per_km"]
            total_cost = calculate_total_freight_cost(shipment, distance, rate)
            print(
                f"Total freight cost for Shipment ID {shipment['id']}: ${total_cost:.2f} using {carrier} ({carrier_rates[carrier]['service_level']})."
            )

            # Estimate delivery time
            delivery_time = estimate_delivery_time(
                distance, carrier_rates[carrier]["speed"]
            )
            print(f"Estimated Delivery Time: {delivery_time.days} days")


# Function to simulate real-time tracking
def simulate_real_time_tracking(shipment):
    print(f"\nReal-time tracking for Shipment ID {shipment['id']}:")
    for i, location in enumerate(shipment["route"]):
        print(
            f"Time: {datetime.now().strftime('%H:%M:%S')}, Location: {location} (Step {i+1}/{len(shipment['route'])})"
        )


# Function to visualize the graph on a real map using folium
def visualize_graph_on_real_map(graph, shipments):
    # Create a base map centered on the world
    m = folium.Map(location=[20, 0], zoom_start=2)

    # Add ports as markers
    for port, data in ports.items():
        folium.Marker(
            location=[data["lat"], data["lon"]],
            popup=f"{port} (Port Fee: ${data['port_fee']})",
            icon=folium.Icon(color="blue"),
        ).add_to(m)

    # Add edges (routes) between ports
    for edge in graph.edges():
        port1, port2 = edge
        coords1 = (ports[port1]["lat"], ports[port1]["lon"])
        coords2 = (ports[port2]["lat"], ports[port2]["lon"])
        folium.PolyLine([coords1, coords2], color="gray", weight=2, opacity=0.7).add_to(
            m
        )

    # Add shipment routes
    for shipment in shipments:
        if "route" in shipment:
            route_coords = [
                (ports[location]["lat"], ports[location]["lon"])
                for location in shipment["route"]
            ]
            folium.PolyLine(
                route_coords,
                color="red",
                weight=3,
                opacity=1,
                tooltip=f"Shipment ID {shipment['id']}",
            ).add_to(m)

    # Save the map to an HTML file
    m.save("real_world_map.html")
    print(
        "Real-world map saved to 'real_world_map.html'. Open this file in a browser to view the map."
    )


# Function to export shipment data to JSON
def export_shipments_to_json(shipments, filename="shipments_output.json"):
    with open(filename, "w") as f:
        json.dump(shipments, f, indent=4)
    print(f"Shipment data exported to {filename}")


# Main function to execute the logistics platform
def main():
    # Create the graph with real-world ports
    graph = create_graph_with_real_world_ports()

    # List available ports
    list_available_ports()

    # List shipping options for a sample shipment
    list_shipping_options("Shanghai", "Rotterdam", weight=100, value=10000)

    # Process shipments (route optimization, cost calculation, and freight audit)
    process_shipments(shipments, graph, carrier_rates)

    # Visualize the graph on a real map
    visualize_graph_on_real_map(graph, shipments)

    # Export shipment data to JSON
    export_shipments_to_json(shipments)


# Run the main function
if __name__ == "__main__":
    main()
