import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
from mpl_toolkits.basemap import Basemap
from concurrent.futures import ThreadPoolExecutor
import time

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_stores = 20
n_dcs = 10
dcs = [f"DC{i}" for i in range(1, n_dcs+1)]
stores = [f"Store{i}" for i in range(1, n_stores+1)]
demands = np.random.randint(50, 200, n_stores)
inventories = np.random.randint(100, 500, n_dcs)

# Assign random coordinates and other data
store_latitudes = np.random.uniform(30, 50, n_stores)
store_longitudes = np.random.uniform(-120, -70, n_stores)
dc_latitudes = np.random.uniform(30, 50, n_dcs)
dc_longitudes = np.random.uniform(-120, -70, n_dcs)

# Create distance matrix
locations = np.column_stack((store_latitudes, store_longitudes))
distances = np.random.uniform(10, 100, (n_dcs, n_stores))
traffic = np.random.uniform(1, 2, (n_dcs, n_stores))
risk = np.random.uniform(1, 1.5, (n_dcs, n_stores))
emissions = np.random.uniform(1, 1.2, (n_dcs, n_stores))
operating_costs = np.random.uniform(50, 200, n_dcs)

class SupplyChainACO:
    def __init__(self, n_ants, n_iterations, alpha, beta, rho, q0):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta    # heuristic importance
        self.rho = rho      # pheromone evaporation rate
        self.q0 = q0        # exploitation vs exploration

    def initialize(self, dcs, stores, distances, traffic, risk, emissions, operating_costs, demands, inventories):
        self.dcs = dcs
        self.stores = stores
        self.distances = distances
        self.traffic = traffic
        self.risk = risk
        self.emissions = emissions
        self.operating_costs = operating_costs
        self.demands = demands
        self.inventories = inventories
        self.n_dcs = len(dcs)
        self.n_stores = len(stores)
        self.pheromones = np.ones((self.n_dcs, self.n_stores))
        operating_costs_broadcast = np.repeat(operating_costs[:, np.newaxis], self.n_stores, axis=1)
        self.heuristic = 1 / (self.distances * self.traffic * self.risk * self.emissions * operating_costs_broadcast)

    def run(self):
        best_solution = None
        best_cost = float('inf')
        for _ in range(self.n_iterations):
            solutions = self.construct_solutions()
            self.update_pheromones(solutions)
            iteration_best = min(solutions, key=lambda x: x[1])
            if iteration_best[1] < best_cost:
                best_solution, best_cost = iteration_best
        return best_solution, best_cost

    def construct_solutions(self):
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.ant_tour, range(self.n_ants)))
        
  
        

    def ant_tour(self, _):
        remaining_demands = self.demands.copy()
        remaining_inventories = self.inventories.copy()
        solution = np.zeros((self.n_dcs, self.n_stores))
        total_cost = 0

        while np.any(remaining_demands > 0):
            for store in range(self.n_stores):
                if remaining_demands[store] > 0:
                    dc = self.select_dc(store, remaining_inventories)
                    if dc is not None:
                        amount = min(remaining_demands[store], remaining_inventories[dc])
                        solution[dc][store] += amount
                        remaining_demands[store] -= amount
                        remaining_inventories[dc] -= amount
                        total_cost += amount * self.calculate_cost(dc, store)

        return solution, total_cost

    def select_dc(self, store, remaining_inventories):
        available_dcs = [dc for dc in range(self.n_dcs) if remaining_inventories[dc] > 0]
        if not available_dcs:
            return None
        if np.random.random() < self.q0:
            probabilities = [self.pheromones[dc][store]**self.alpha * self.heuristic[dc][store]**self.beta for dc in available_dcs]
            return available_dcs[np.argmax(probabilities)]
        else:
            probabilities = [self.pheromones[dc][store]**self.alpha * self.heuristic[dc][store]**self.beta for dc in available_dcs]
            probabilities = probabilities / np.sum(probabilities)
            return np.random.choice(available_dcs, p=probabilities)

    def calculate_cost(self, dc, store):
        return (self.distances[dc][store] * self.traffic[dc][store] * 
                self.risk[dc][store] * self.emissions[dc][store] * 
                self.operating_costs[dc])

    def update_pheromones(self, solutions):
        self.pheromones *= (1 - self.rho)
        for solution, cost in solutions:
            self.pheromones += solution / cost

    def visualize_solution(self, solution, cost):
        G = nx.DiGraph()
        G.add_nodes_from([f"DC{i}" for i in range(self.n_dcs)], bipartite=0)
        G.add_nodes_from([f"Store{i}" for i in range(self.n_stores)], bipartite=1)
        pos = nx.bipartite_layout(G, [f"DC{i}" for i in range(self.n_dcs)])

        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)

        labels = {}
        for i in range(self.n_dcs):
            labels[f"DC{i}"] = f"DC{i}\nInv: {self.inventories[i]}"
        for i in range(self.n_stores):
            labels[f"Store{i}"] = f"Store{i}\nDem: {self.demands[i]}"
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax)

        paths = []
        for dc in range(self.n_dcs):
            for store in range(self.n_stores):
                if solution[dc][store] > 0:
                    G.add_edge(f"DC{dc}", f"Store{store}", weight=solution[dc][store])
                    nx.draw_networkx_edges(G, pos, edgelist=[(f"DC{dc}", f"Store{store}")], 
                                            edge_color='r', arrows=True, ax=ax)
                    nx.draw_networkx_edge_labels(G, pos, edge_labels={(f"DC{dc}", f"Store{store}"): f"{solution[dc][store]:.1f}"}, ax=ax)
                    paths.append(f"DC{dc} -> Store{store}: {solution[dc][store]:.1f} units")
                else:
                    G.add_edge(f"DC{dc}", f"Store{store}")
                    nx.draw_networkx_edges(G, pos, edgelist=[(f"DC{dc}", f"Store{store}")], 
                                            edge_color='black', style='dotted', arrows=True, ax=ax)

        ax.set_title(f"Optimal Supply Chain Paths (Total Cost: {cost:.2f})")
        ax.axis('off')
        plt.tight_layout()

        st.pyplot(fig)
        st.write("##### Optimal distribution paths:")
        for path in paths:
            st.write(f"- {path}")

def plot_demand_distribution():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a Basemap instance with zoomed-in parameters
    m = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=60,
                llcrnrlon=-150, urcrnrlon=-60, resolution='l', ax=ax)
    
    # Draw map boundaries and fill the continents
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawcoastlines()
    m.drawcountries()
    
    # Convert lat/lon to map projection coordinates
    x_stores, y_stores = m(store_longitudes, store_latitudes)
    
    # Scatter plot of stores with demand color coding
    scatter = m.scatter(x_stores, y_stores, c=demands, cmap='viridis', s=50, edgecolor='k', label='Stores')
    
    # Add labels for each store
    for i, txt in enumerate(stores):
        plt.text(x_stores[i], y_stores[i], txt, fontsize=8, color='black', ha='left', va='bottom')
    
    plt.title("Demand Distribution by Store on World Map")
    plt.colorbar(scatter, label="Demand")
    
    return fig

def plot_inventory_distribution():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a Basemap instance with zoomed-in parameters
    m = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=60,
                llcrnrlon=-150, urcrnrlon=-60, resolution='l', ax=ax)
    
    # Draw map boundaries and fill the continents
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawcoastlines()
    m.drawcountries()
    
    # Convert lat/lon to map projection coordinates
    x_dcs, y_dcs = m(dc_longitudes, dc_latitudes)
    
    # Scatter plot of distribution centers with inventory color coding
    scatter = m.scatter(x_dcs, y_dcs, c=inventories, cmap='plasma', s=100, edgecolor='k', label='Distribution Centers')
    
    # Add labels for each distribution center
    for i, txt in enumerate(dcs):
        plt.text(x_dcs[i], y_dcs[i], txt, fontsize=8, color='black', ha='left', va='bottom')
    
    plt.title("Inventory Distribution by Distribution Center on World Map")
    plt.colorbar(scatter, label="Inventory")
    
    return fig

def calculate_risk_statistics(risk):
    flat_risk = risk.flatten()
    highest_risk = np.max(flat_risk)
    lowest_risk = np.min(flat_risk)
    avg_risk = np.mean(flat_risk)
    median_risk = np.median(flat_risk)
    std_dev_risk = np.std(flat_risk)
    
    highest_risk_route = np.unravel_index(np.argmax(risk), risk.shape)
    lowest_risk_route = np.unravel_index(np.argmin(risk), risk.shape)
    
    return {
        "Highest Risk Route": f"DC{highest_risk_route[0]} to Store{highest_risk_route[1]} (Risk: {highest_risk:.2f})",
        "Lowest Risk Route": f"DC{lowest_risk_route[0]} to Store{lowest_risk_route[1]} (Risk: {lowest_risk:.2f})",
        "Average Risk": f"{avg_risk:.2f}",
        "Median Risk": f"{median_risk:.2f}",
        "Standard Deviation of Risk": f"{std_dev_risk:.2f}",
        "Risk Range": f"{lowest_risk:.2f} - {highest_risk:.2f}"
    }

def plot_risk_graph():
    G = nx.Graph()
    for i in range(n_dcs):
        for j in range(n_stores):
            G.add_edge(f"DC{i}", f"Store{j}", weight=risk[i][j])

    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = nx.draw_networkx_edges(G, pos, edge_color=[G[u][v]['weight'] for u, v in G.edges()],
                                   edge_cmap=plt.cm.YlOrRd, width=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    plt.colorbar(edges, label='Risk')
    ax.set_title("Risk Graph")
    ax.axis('off')
    
    stats = calculate_risk_statistics(risk)
    return fig, stats

def calculate_traffic_statistics(traffic):
        flat_traffic = traffic.flatten()
        most_congested = np.max(flat_traffic)
        least_congested = np.min(flat_traffic)
        avg_congestion = np.mean(flat_traffic)
        median_congestion = np.median(flat_traffic)
        std_dev_congestion = np.std(flat_traffic)
    
        most_congested_route = np.unravel_index(np.argmax(traffic), traffic.shape)
        least_congested_route = np.unravel_index(np.argmin(traffic), traffic.shape)
    
        return {
        "Most Congested Route": f"DC{most_congested_route[0]} to Store{most_congested_route[1]} (Congestion: {most_congested:.2f})",
        "Least Congested Route": f"DC{least_congested_route[0]} to Store{least_congested_route[1]} (Congestion: {least_congested:.2f})",
        "Average Congestion": f"{avg_congestion:.2f}",
        "Median Congestion": f"{median_congestion:.2f}",
        "Standard Deviation of Congestion": f"{std_dev_congestion:.2f}",
        "Congestion Range": f"{least_congested:.2f} - {most_congested:.2f}"
        }
    

def plot_traffic_graph():
    G = nx.Graph()
    for i in range(n_dcs):
        for j in range(n_stores):
            G.add_edge(f"DC{i}", f"Store{j}", weight=traffic[i][j])

    pos = nx.spring_layout(G)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = nx.draw_networkx_edges(G, pos, edge_color=[G[u][v]['weight'] for u, v in G.edges()],
                                   edge_cmap=plt.cm.YlOrRd, width=2, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    plt.colorbar(edges, label='Traffic')
    ax.set_title("Traffic Graph")
    ax.axis('off')
    
    stats = calculate_traffic_statistics(traffic)
    return fig, stats

def plot_initial_setup():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a Basemap instance with zoomed-in parameters
    m = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=60,
                llcrnrlon=-150, urcrnrlon=-60, resolution='l', ax=ax)
    
    # Draw map boundaries and fill the continents
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawcoastlines()
    m.drawcountries()
    
    # Convert lat/lon to map projection coordinates
    x_stores, y_stores = m(store_longitudes, store_latitudes)
    x_dcs, y_dcs = m(dc_longitudes, dc_latitudes)
    
    # Plot stores
    scatter_stores = m.scatter(x_stores, y_stores, c='blue', s=50, label='Stores', edgecolor='k')
    for i, txt in enumerate(stores):
        plt.text(x_stores[i], y_stores[i], txt, fontsize=8, color='black', ha='left', va='bottom')
    
    # Plot distribution centers
    scatter_dcs = m.scatter(x_dcs, y_dcs, c='red', s=100, label='Distribution Centers', edgecolor='k')
    for i, txt in enumerate(dcs):
        plt.text(x_dcs[i], y_dcs[i], txt, fontsize=8, color='black', ha='left', va='bottom')
    
    plt.title("Initial Setup of Stores and Distribution Centers on World Map")
    plt.legend()
    
    return fig

def calculate_recommended_locations():
    # Determine optimal number of clusters using the Elbow Method
    distortions = []
    K = range(1, 11)  # Try different numbers of clusters
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(locations)
        distortions.append(kmeans.inertia_)
    
    # Find the optimal number of clusters (Elbow point)
    optimal_k = K[np.argmin(np.diff(distortions, 2)) + 1]  # Approximate elbow point
    
    # Perform KMeans with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(locations)

    weighted_centroids = []
    for i in range(optimal_k):
        cluster_points = locations[cluster_labels == i]
        cluster_demands = demands[cluster_labels == i]
        weighted_centroid = np.average(cluster_points, axis=0, weights=cluster_demands)
        weighted_centroids.append(weighted_centroid)

    weighted_centroids = np.array(weighted_centroids)

    # Plot 1: Old Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(store_longitudes, store_latitudes, c=cluster_labels, cmap='viridis', alpha=0.7, s=demands, edgecolor='k')
    ax.scatter(weighted_centroids[:, 1], weighted_centroids[:, 0], c='red', marker='x', s=200, linewidths=3, label='Recommended DC Locations')
    plt.colorbar(scatter, label='Cluster')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Store Clusters and Recommended Distribution Centers')
    ax.legend()

    for i, centroid in enumerate(weighted_centroids):
        cluster_points = locations[cluster_labels == i]
        for point in cluster_points:
            ax.plot([centroid[1], point[1]], [centroid[0], point[0]], 'k-', alpha=0.1)

    # Plot 2: World Map
    fig_map, ax_map = plt.subplots(figsize=(12, 8))
    m = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=60,
                llcrnrlon=-150, urcrnrlon=-60, resolution='l', ax=ax_map)
    
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    m.drawcoastlines()
    m.drawcountries()
    
    x_stores, y_stores = m(store_longitudes, store_latitudes)
    x_centroids, y_centroids = m(weighted_centroids[:, 1], weighted_centroids[:, 0])
    
    m.scatter(x_centroids, y_centroids, c='red', marker='x', s=200, linewidths=3, label='Recommended DC Locations')
    
    for i, centroid in enumerate(weighted_centroids):
        plt.text(x_centroids[i], y_centroids[i], f'RC{i+1}', fontsize=8, color='black', ha='right', va='bottom')

    plt.title('Store Clusters and Recommended Distribution Centers on World Map')
    plt.legend()

    for i, centroid in enumerate(weighted_centroids):
        cluster_points = locations[cluster_labels == i]
        for point in cluster_points:
            ax_map.plot([centroid[1], point[1]], [centroid[0], point[0]], 'k-', alpha=0.1)

    return fig, fig_map, weighted_centroids

def plot_cost_breakdown(cost_components):
    fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size for better readability
    
    labels = cost_components.keys()
    sizes = cost_components.values()
    
    # Create the pie chart with adjustments
    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct='',
        startangle=140,
        pctdistance=0.85,  # Distance of percentage text from the center
        labeldistance=1.1  # Distance of labels from the center
    )
    
    # Improve readability by adjusting the font size of the percentage text and labels
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(12)
    
    ax.set_title("Cost Breakdown")
    
    # Create custom legend entries that include both labels and values
    legend_entries = [f'{label}: ${value:,.2f}' for label, value in zip(labels, sizes)]
    
    # Add a legend with custom entries
    ax.legend(wedges, legend_entries, title="Cost Components", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    return fig

def plot_demand_fulfillment(solution):
    fig, ax = plt.subplots()
    for dc in range(n_dcs):
        for store in range(n_stores):
            if solution[dc][store] > 0:
                ax.plot([store_longitudes[dc], store_longitudes[store]],
                        [store_latitudes[dc], store_latitudes[store]],
                        'r-', alpha=0.5)
    ax.set_title("Demand Fulfillment Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig

def plot_store_clustering():
    # Determine optimal number of clusters using the Elbow Method
    distortions = []
    K = range(1, 11)  # Try different numbers of clusters
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(locations)
        distortions.append(kmeans.inertia_)
    
    # Find the optimal number of clusters (Elbow point)
    optimal_k = K[np.argmin(np.diff(distortions, 2)) + 1]  # Approximate elbow point
    kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(locations)
    labels = kmeans.labels_

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scale the sizes of the dots based on demand
    # Adjust the size scaling factor as needed (e.g., 10) to make the dots more or less prominent
    size_scaling_factor = 2
    sizes = demands * size_scaling_factor
    
    scatter = ax.scatter(store_longitudes, store_latitudes, c=labels, cmap='viridis', alpha=0.7, s=sizes)
    plt.colorbar(scatter, label='Cluster')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Store Clustering Based on Demand')
    
    return fig

def plot_operational_cost_trends(costs):
    fig, ax = plt.subplots()
    ax.plot(costs, marker='o', linestyle='-')
    ax.set_title("Operational Cost Trends")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    return fig

def plot_capacity_utilization():
    utilizations = [1 - (inv - np.sum(solution[dc, :]) / inv) for dc, inv in enumerate(inventories)]
    fig, ax = plt.subplots()
    ax.bar(range(n_dcs), utilizations, color='orange')
    ax.set_title("Capacity Utilization of Distribution Centers")
    ax.set_xlabel("Distribution Center")
    ax.set_ylabel("Utilization")
    return fig

def main():
    st.title("SCAIO Dashboard")
    
    # Parameters
    st.sidebar.write("### Configure Hyperparameters")
    n_ants = st.sidebar.slider("Number of Ants", 10, 100, 50)
    n_iterations = st.sidebar.slider("Number of Iterations", 10, 200, 100)
    alpha = st.sidebar.slider("Alpha (pheromone importance)", 0.1, 5.0, 1.0)
    beta = st.sidebar.slider("Beta (heuristic importance)", 0.1, 5.0, 2.0)
    rho = st.sidebar.slider("Pheromone Evaporation Rate", 0.01, 1.0, 0.1)
    q0 = st.sidebar.slider("q0 (exploitation vs exploration)", 0.0, 1.0, 0.9)

    tab1, tab2, tab3, tab4 = st.tabs(["Network Setup", "Traffic Analysis", "Risk Analysis", "Optimize"])
    
    # Visualization
    with tab1:
        with st.expander("##### Supply Chain Setup", expanded=True):
            st.pyplot(plot_initial_setup()) 
        
        with st.expander("##### Demand Distribution", expanded=True):
            st.pyplot(plot_demand_distribution())
        
        with st.expander("##### Inventory Distribution", expanded=True):
            st.pyplot(plot_inventory_distribution())
        
        with st.expander("##### Store Clustering", expanded=True):
            st.pyplot(plot_store_clustering())

        with st.expander("##### Cost Breakdown", expanded=True):
            cost_components = {
                'Distance': np.sum(distances),
                'Traffic': np.sum(traffic),
                'Risk': np.sum(risk),
                'Emissions': np.sum(emissions),
                'Operating Costs': np.sum(operating_costs)
            }
            st.pyplot(plot_cost_breakdown(cost_components))

    with tab2:
        with st.expander("##### Traffic Graph", expanded=True):
            fig, traffic_stats = plot_traffic_graph()
            st.pyplot(fig)
            
            st.write("##### Traffic Statistics:")
            for stat, value in traffic_stats.items():
                st.write(f"- **{stat}:** {value}")

    with tab3:
        with st.expander("##### Risk Graph", expanded=True):
            fig, risk_stats = plot_risk_graph()
            st.pyplot(fig)
            
            st.write("##### Risk Statistics:")
            for stat, value in risk_stats.items():
                st.write(f"- **{stat}:** {value}")

    with tab4:
        if (st.button("Run Optimizations")):
            with st.expander("##### Relocation Recommendations"):
                fig, fig_map, recommended_locations = calculate_recommended_locations()
                st.pyplot(fig)
                st.pyplot(fig_map)
                st.write("##### Recommended DC Locations:")
                for i, location in enumerate(recommended_locations):
                    st.write(f"- DC {i+1}: Latitude {location[0]:.4f}, Longitude {location[1]:.4f}")

            with st.expander("##### Optimized Distribution Paths"):
                aco = SupplyChainACO(n_ants, n_iterations, alpha, beta, rho, q0)
                aco.initialize(dcs, stores, distances, traffic, risk, emissions, operating_costs, demands, inventories)
                best_solution, best_cost = aco.run()
                aco.visualize_solution(best_solution, best_cost)

                st.write("###### Demand Fulfillment Map")
                st.pyplot(plot_demand_fulfillment(np.zeros((n_dcs, n_stores))))
        

if __name__ == "__main__":
    main()