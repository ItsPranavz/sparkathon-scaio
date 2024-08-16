import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
from mpl_toolkits.basemap import Basemap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time

def generate_seasonal_data(n_periods, n_dcs, trend_range=(50, 150), seasonal_amplitude=10, noise_scale=5):

    data = {}

    for dc_id in range(1, n_dcs + 1):
        dc_name = f"DC{dc_id}"
        
        # Generate a linear trend component
        trend = np.linspace(trend_range[0], trend_range[1], n_periods)
        
        # Generate a seasonal component (e.g., sine wave for periodicity)
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_periods) / (n_periods / 12))
        
        # Generate random noise
        noise = np.random.normal(scale=noise_scale, size=n_periods)
        
        # Combine trend, seasonal, and noise
        demand = trend + seasonal + noise
        
        # Add to the dictionary
        data[dc_name] = demand.tolist()
    
    return data

# Set random seed for reproducibility
np.random.seed(12)

# Generate sample data
n_stores = 10
n_dcs = 5
dcs = [f"DC{i}" for i in range(1, n_dcs+1)]
stores = [f"Store{i}" for i in range(1, n_stores+1)]
demands = np.random.randint(50, 200, n_stores)
inventories = np.random.randint(100, 500, n_dcs)

# Assign random coordinates and other data
store_latitudes = np.random.uniform(15.0, 29.0, n_stores)
store_longitudes = np.random.uniform(73.0, 86.0, n_stores)
dc_latitudes = np.random.uniform(15.0, 29.0, n_dcs)
dc_longitudes = np.random.uniform(73.0, 86.0, n_dcs)

# Create distance matrix
locations = np.column_stack((store_latitudes, store_longitudes))
distances = np.random.uniform(10, 100, (n_dcs, n_stores))
traffic = np.random.uniform(1, 2, (n_dcs, n_stores))
risk = np.random.uniform(1, 1.5, (n_dcs, n_stores))
emissions = np.random.uniform(1, 1.2, (n_dcs, n_stores))
operating_costs = np.random.uniform(50, 200, n_dcs)

def top_3_dcs_with_highest_risks(risk, dc_latitudes, dc_longitudes):
    # Calculate the sum of risks for each DC
    total_risks = np.sum(risk, axis=1)
    
    # Get the indices of the DCs sorted by total risk in descending order
    sorted_indices = np.argsort(total_risks)[::-1]
    
    # Get the top 3 DCs
    top_3_indices = sorted_indices[:3]
    
    # Create a DataFrame with details of the top 3 DCs
    dc_names = [f"DC{i}" for i in top_3_indices]  # Example names for DCs
    top_3_details = {
        "DC Name": dc_names,
        "Latitude": dc_latitudes[top_3_indices],
        "Longitude": dc_longitudes[top_3_indices],
        "Total Risk": total_risks[top_3_indices]
    }
    
    df_top_3 = pd.DataFrame(top_3_details)
    
    return df_top_3

# Function to find nearest industrial area
def find_nearest_industrial_area(centroid, df):
    distances = cdist([centroid], df[['Latitude', 'Longitude']])
    nearest_index = distances.argmin()
    return df.iloc[nearest_index]

# Function to find optimal locations
def find_optimal_locations(G, num_optimal=3):
    cost_matrix = []
    for node in G.nodes():
        total_cost = (G.nodes[node]['land_cost'] + 
                      G.nodes[node]['labor_cost'] * 30 +
                      G.nodes[node]['electricity_cost'] * 1000 +
                      G.nodes[node]['risk_factor'] * 1000)
        cost_matrix.append((node, total_cost))
    return sorted(cost_matrix, key=lambda x: x[1])[:num_optimal]

# Function to forecast demand for a single store using SARIMAX
def forecast_store_demand(demand_history, forecast_days=10):
    # Convert to DataFrame
    df = pd.DataFrame({'demand': demand_history})
    df.index = pd.date_range(end='2023-08-16', periods=len(demand_history), freq='D')
    
    # Split the data
    train = df[:-forecast_days]
    test = df[-forecast_days:]
    
    # Fit SARIMAX model
    model = SARIMAX(train['demand'], order=(1,1,1), seasonal_order=(1,1,1,12))
    results = model.fit()
    
    # Forecast
    forecast = results.get_forecast(steps=forecast_days)
    forecast_mean = forecast.predicted_mean
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test['demand'], forecast_mean))
    
    return forecast_mean.tolist(), rmse

# Function to forecast demands for all stores in parallel
def forecast_all_stores(input_data, forecast_days=10):
    forecasts = {}
    avg_forecasts = {}
    rmse_values = {}
    
    def process_store(store, demand_history):
        forecast, rmse = forecast_store_demand(demand_history, forecast_days)
        return store, forecast, np.mean(forecast), rmse
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda item: process_store(item[0], item[1]), input_data.items()))
    
    for store, forecast, avg_forecast, rmse in results:
        forecasts[store] = forecast
        avg_forecasts[store] = avg_forecast
        rmse_values[store] = rmse
    
    return forecasts, avg_forecasts, rmse_values

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

    def plot_capacity_utilization(self):
        utilizations = [1 - (inv - np.sum(solution[dc, :]) / inv) for dc, inv in enumerate(inventories)]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(range(n_dcs), utilizations, color='orange')
        ax.set_title("Capacity Utilization of Distribution Centers")
        ax.set_xlabel("Distribution Center")
        ax.set_ylabel("Utilization")
        return fig

def calculate_supply_chain_costs(distances, traffic, risk, emissions, operating_costs, demands, inventories):
  
    transportation_costs = np.sum(distances * traffic * demands) * 0.0357
    
    risk_costs = np.sum(risk * demands) * 0.44
 
    emission_costs = np.sum(emissions * demands) * 0.36
   
    total_operating_costs = np.sum(operating_costs) * 12.61
   
    holding_cost_rate = 0.2
    average_product_value = 50  # Assuming an average product value of $50
    holding_costs = np.sum(inventories) * average_product_value * holding_cost_rate * 0.42
    
    total_cost = transportation_costs + risk_costs + emission_costs + total_operating_costs + holding_costs
    
    return {
        "Transportation Costs": transportation_costs,
        "Risk-related Costs": risk_costs,
        "Emission Costs": emission_costs,
        "Operating Costs": total_operating_costs,
        "Holding Costs": holding_costs,
        "Total Supply Chain Cost": total_cost
    }

def plot_demand_distribution():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a Basemap instance with zoomed-in parameters
    m = Basemap(projection='merc', llcrnrlat=8, urcrnrlat=35, llcrnrlon=68, urcrnrlon=97, resolution='i', ax=ax)
    m.drawcountries(linewidth=1.0, color='black')
    m.fillcontinents(color='lightgreen', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    m.drawstates(linewidth=0.5, color='gray')
    
    # Convert lat/lon to map projection coordinates
    x_stores, y_stores = m(store_longitudes, store_latitudes)
    
    # Scatter plot of stores with demand color coding
    scatter = m.scatter(x_stores, y_stores, c=demands, cmap='YlOrRd', s=50, edgecolor='k', label='Stores')
    
    # Add labels for each store
    for i, txt in enumerate(stores):
        plt.text(x_stores[i], y_stores[i], txt, fontsize=8, color='black', ha='left', va='bottom')
    
    plt.title("Demand Distribution by Store on World Map")
    plt.colorbar(scatter, label="Demand")
    
    return fig

def plot_inventory_distribution():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a Basemap instance with zoomed-in parameters
    m = Basemap(projection='merc', llcrnrlat=8, urcrnrlat=35, llcrnrlon=68, urcrnrlon=97, resolution='i', ax=ax)
    m.drawcountries(linewidth=1.0, color='black')
    m.fillcontinents(color='lightgreen', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    m.drawstates(linewidth=0.5, color='gray')
    
    # Convert lat/lon to map projection coordinates
    x_dcs, y_dcs = m(dc_longitudes, dc_latitudes)
    
    # Scatter plot of distribution centers with inventory color coding
    scatter = m.scatter(x_dcs, y_dcs, c=inventories, cmap='YlOrRd', s=100, edgecolor='k', label='Distribution Centers')
    
    # Add labels for each distribution center
    for i, txt in enumerate(dcs):
        plt.text(x_dcs[i], y_dcs[i], txt, fontsize=8, color='black', ha='left', va='bottom')
    
    plt.title("Inventory Distribution by Distribution Center on World Map")
    plt.colorbar(scatter, label="Inventory")
    
    return fig

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
    m = Basemap(projection='merc', llcrnrlat=8, urcrnrlat=35, llcrnrlon=68, urcrnrlon=97, resolution='i', ax=ax)
    m.drawcountries(linewidth=1.0, color='black')
    m.fillcontinents(color='lightgreen', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    m.drawstates(linewidth=0.5, color='gray')
    
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
    m = Basemap(projection='merc', llcrnrlat=8, urcrnrlat=35, llcrnrlon=68, urcrnrlon=97, resolution='i', ax=ax)
    m.drawcountries(linewidth=1.0, color='black')
    m.fillcontinents(color='lightgreen', lake_color='aqua')
    m.drawmapboundary(fill_color='aqua')
    m.drawstates(linewidth=0.5, color='gray')
    
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

    return fig, fig_map, weighted_centroids, kmeans.labels_, optimal_k

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

def main():
    st.title("SCAIO Dashboard: AI-Powered Supply Chain Optimization")
    st.markdown("*Visualize, analyze, and optimize your supply chain network with advanced AI techniques.*")

    # Parameters
    st.sidebar.write("### Configure Optimization")
    st.sidebar.markdown("*Adjust these parameters to fine-tune the optimization algorithm.*")
    n_ants = st.sidebar.slider("Number of Ants", 10, 100, 50, help="More ants can improve solution quality but increase computation time.")
    n_iterations = st.sidebar.slider("Number of Iterations", 10, 200, 100, help="More iterations may lead to better solutions but take longer to compute.")
    alpha = st.sidebar.slider("Alpha (pheromone importance)", 0.1, 5.0, 1.0, help="Higher values give more importance to pheromone trails.")
    beta = st.sidebar.slider("Beta (heuristic importance)", 0.1, 5.0, 2.0, help="Higher values give more importance to heuristic information.")
    rho = st.sidebar.slider("Pheromone Evaporation Rate", 0.01, 1.0, 0.1, help="Controls how quickly pheromone trails decay.")
    q0 = st.sidebar.slider("q0 (exploitation vs exploration)", 0.0, 1.0, 0.9, help="Higher values favor exploitation of known good paths.")

    st.sidebar.write("### Configure Forecasting")
    st.sidebar.markdown("*Adjust these parameters to fine-tune the forecasting algorithm.*")
    n_periods = st.sidebar.slider("Number of periods", 10, 100, 50, help="Number of periods to consider for training forecasting model.")
    n_forecast_days = st.sidebar.slider("Forecasting length", 5, 30
    , 10, help="Number of days to forecast from current date.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Network Setup", "Traffic Analysis", "Risk Analysis", "Optimize", "Forecast"])
    
    with tab1:
        with st.spinner("Loading visualizations... please wait!"):
            st.markdown("### Network Setup: Visualize Your Supply Chain")
            
            with st.expander("Supply Chain Setup", expanded=True):
                st.markdown("*Overview of distribution centers and stores on a map.*")
                st.pyplot(plot_initial_setup()) 
            
            with st.expander("Demand Distribution", expanded=True):
                st.markdown("*Visualizes store locations with demand represented by color intensity.*")
                st.pyplot(plot_demand_distribution())
            
            with st.expander("Inventory Distribution", expanded=True):
                st.markdown("*Shows distribution center locations with inventory levels indicated by color.*")
                st.pyplot(plot_inventory_distribution())
            
            with st.expander("Store Clustering", expanded=True):
                st.markdown("*Groups stores based on location and demand using K-means clustering.*")
                st.pyplot(plot_store_clustering())

            with st.expander("Cost Breakdown", expanded=True):
                supply_chain_costs = calculate_supply_chain_costs(distances, traffic, risk, emissions, operating_costs, demands, inventories)
                
                st.write("##### Estimated Supply Chain Costs:")
                total_cost = supply_chain_costs["Total Supply Chain Cost"]
                for cost_type, cost_value in supply_chain_costs.items():
                        if cost_type != "Total Supply Chain Cost":
                            percentage = (cost_value / total_cost) * 100
                            st.write(f"- **{cost_type}:** ${cost_value:,.2f} ({percentage:.2f}%)")


        
                fig, ax = plt.subplots(figsize=(10, 8))
                cost_types = list(supply_chain_costs.keys())[:-1]  # Exclude total cost
                cost_values = [supply_chain_costs[cost_type] for cost_type in cost_types]
                
                wedges, texts = ax.pie(cost_values, startangle=90)
        
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig.gca().add_artist(centre_circle)
                
                ax.axis('equal') 
                ax.set_title("Supply Chain Cost Breakdown")
            
                ax.legend(wedges, cost_types,
                        title="Cost Components",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1))
                
                plt.tight_layout()
                st.pyplot(fig)

    with tab2:
        with st.spinner("Analyzing traffic... please wait!"):
            st.markdown("### Traffic Analysis: Understand Network Congestion")
            with st.expander("Traffic Graph", expanded=True):
                st.markdown("*Visualizes traffic conditions between distribution centers and stores.*")
                fig, traffic_stats = plot_traffic_graph()
                st.pyplot(fig)
                
                st.write("##### Traffic Statistics:")
                for stat, value in traffic_stats.items():
                    st.write(f"- **{stat}:** {value}")

    with tab3:
        with st.spinner("Analyzing risk... please wait!"):
            df_top_3 = top_3_dcs_with_highest_risks(risk, dc_latitudes, dc_longitudes)
            st.markdown("### Risk Analysis: Identify Vulnerable Links")
            with st.expander("Risk Graph", expanded=True):
                st.markdown("*Highlights high-risk connections in the supply chain network.*")
                fig, risk_stats = plot_risk_graph()
                st.pyplot(fig)
                
                st.write("##### Risk Statistics:")
                for stat, value in risk_stats.items():
                    st.write(f"- **{stat}:** {value}")

            with st.expander("### Vulnerability Assessment", expanded=True):
                st.markdown("*Shows the high risk DCs and alternatives for relocation*")
                st.write("##### High Risk DCs")
                st.table(df_top_3)

                coordinates_array = df_top_3[['Latitude', 'Longitude']].values.tolist()

                st.markdown("*Suggests optimal locations for new distribution centers based on demand patterns.*")

                # Load dataset
                @st.cache_data
                def load_data():
                    return pd.read_csv("dataset.csv")

                df = load_data()

                # Find nearest industrial areas to centroids
                nearest_areas = [find_nearest_industrial_area(coord, df) for coord in coordinates_array]
                cumulative_demands = [100 for _ in range(len(nearest_areas))]

                # Create a supply chain network
                G = nx.Graph()

                # Add nodes (nearest industrial areas) to the graph
                for i, area in enumerate(nearest_areas):
                    demand = cumulative_demands[i]
                    land_cost = area['Land Cost (INR/sq ft)'] * demand * 0.1
                    labor_cost = area['Labour Cost (INR/day)'] * demand * 0.05
                    electricity_cost = area['Electricity Cost (INR/unit)'] * demand * 0.02
                    risk_factor = 1 + (demand / max(cumulative_demands)) * 4

                    G.add_node(area['Industrial Area'], 
                            land_cost=land_cost,
                            labor_cost=labor_cost,
                            electricity_cost=electricity_cost,
                            risk_factor=risk_factor,
                            lat=area['Latitude'],
                            lon=area['Longitude'],
                            demand=demand)

                # Find optimal locations
                optimal_locations = find_optimal_locations(G, num_optimal=1)

                # Visualize the network with emphasis on the optimal locations
                st.write("##### Map of recommended DC relocations for reduced risk")
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Reuse the Basemap for network visualization
                m = Basemap(projection='merc', llcrnrlat=8, urcrnrlat=35, llcrnrlon=68, urcrnrlon=97, resolution='i', ax=ax)
                m.drawcountries(linewidth=1.0, color='black')
                m.fillcontinents(color='lightgreen', lake_color='aqua')
                m.drawmapboundary(fill_color='aqua')
                m.drawstates(linewidth=0.5, color='gray')

                pos = {node: m(G.nodes[node]['lon'], G.nodes[node]['lat']) for node in G.nodes()}
                nx.draw(G, pos, node_color='lightblue', node_size=500, with_labels=True, ax=ax)

                # Plot optimal locations
                colors = plt.cm.rainbow(np.linspace(0, 1, len(optimal_locations)))
                for i, (location, _) in enumerate(optimal_locations):
                    nx.draw_networkx_nodes(G, pos, nodelist=[location], node_color=[colors[i]], node_size=700, ax=ax)
                    ax.annotate(f'Optimal {i+1}', pos[location], 
                                xytext=(5, 5), textcoords='offset points')

                # Plot high-risk DCs with cross marker
                high_risk_pos = {row['DC Name']: m(row['Longitude'], row['Latitude']) for index, row in df_top_3.iterrows()}
                nx.draw_networkx_nodes(G, high_risk_pos, nodelist=df_top_3['DC Name'], node_color='red', node_shape='x', node_size=700, ax=ax)

                # Add a legend
                legend_labels = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='New Suggested DC'),
                    plt.Line2D([0], [0], marker='x', color='r', markerfacecolor='red', markersize=10, label='Current High Risk DC')
                ]
                ax.legend(handles=legend_labels, loc='upper right')

                ax.set_title("Supply Chain Network - Optimal Locations")
                st.pyplot(fig)

                # Display details for all recommended nodes
                st.write("##### Estimated setup costs and risk levels")

                details = []
                for node in G.nodes():
                    # Calculate the total cost for the node
                    total_cost = (G.nodes[node]['land_cost'] + 
                                G.nodes[node]['labor_cost'] * 30 + 
                                G.nodes[node]['electricity_cost'] * 1000 + 
                                G.nodes[node]['risk_factor'] * 1000)

                    # Append the node details to the list
                    details.append({
                        "Industrial Area": node,
                        "Latitude": f"{G.nodes[node]['lat']:.4f}",
                        "Longitude": f"{G.nodes[node]['lon']:.4f}",
                        "Total Cost (INR)": f"{total_cost:.2f}",
                        "Risk Factor": f"{G.nodes[node]['risk_factor']:.2f}"
                    })
                
                # Convert to DataFrame and display as table
                df_details = pd.DataFrame(details)
                st.dataframe(df_details)

    with tab4:
        st.markdown("### Optimize: Improve Your Supply Chain")
        with st.spinner("Optimizing... This may take a moment."):
            if st.button("Run Optimizations"):
                with st.expander("Relocation Recommendations", expanded=True):
                    # Load dataset
                    @st.cache_data
                    def load_data():
                        return pd.read_csv("dataset.csv")

                    df = load_data()

                    fig, fig_map, centroids, cluster_labels, n_clusters = calculate_recommended_locations()

                    # Calculate cumulative demands for each centroid
                    cumulative_demands = [np.sum(demands[cluster_labels == i]) for i in range(n_clusters)]

                    # Find nearest industrial areas to centroids
                    nearest_areas = [find_nearest_industrial_area(centroid, df) for centroid in centroids]

                    # Create a supply chain network
                    G = nx.Graph()

                    # Add nodes (nearest industrial areas) to the graph
                    for i, area in enumerate(nearest_areas):
                        demand = cumulative_demands[i]
                        land_cost = area['Land Cost (INR/sq ft)'] * demand * 0.1
                        labor_cost = area['Labour Cost (INR/day)'] * demand * 0.05
                        electricity_cost = area['Electricity Cost (INR/unit)'] * demand * 0.02
                        risk_factor = 1 + (demand / max(cumulative_demands)) * 4

                        G.add_node(area['Industrial Area'], 
                                land_cost=land_cost,
                                labor_cost=labor_cost,
                                electricity_cost=electricity_cost,
                                risk_factor=risk_factor,
                                lat=area['Latitude'],
                                lon=area['Longitude'],
                                demand=demand)

                    # Find optimal locations
                    optimal_locations = find_optimal_locations(G, num_optimal=3)

                    # Visualize the network with emphasis on the optimal locations
                    st.write("##### Map of recommended DC locations")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Reuse the Basemap for network visualization
                    m = Basemap(projection='merc', llcrnrlat=8, urcrnrlat=35, llcrnrlon=68, urcrnrlon=97, resolution='i', ax=ax)
                    m.drawcountries(linewidth=1.0, color='black')
                    m.fillcontinents(color='lightgreen', lake_color='aqua')
                    m.drawmapboundary(fill_color='aqua')
                    m.drawstates(linewidth=0.5, color='gray')

                    pos = {node: m(G.nodes[node]['lon'], G.nodes[node]['lat']) for node in G.nodes()}
                    nx.draw(G, pos, node_color='lightblue', node_size=500, with_labels=True, ax=ax)

                    colors = plt.cm.rainbow(np.linspace(0, 1, len(optimal_locations)))
                    for i, (location, _) in enumerate(optimal_locations):
                        nx.draw_networkx_nodes(G, pos, nodelist=[location], node_color=[colors[i]], node_size=700, ax=ax)
                        ax.annotate(f'Optimal {i+1}', pos[location], 
                                    xytext=(5, 5), textcoords='offset points')

                    ax.set_title("Supply Chain Network - Optimal Locations")
                    st.pyplot(fig)

                    # Display details for all recommended nodes
                    st.write("##### Estimated setup costs")

                    details = []
                    for node in G.nodes():
                        # Calculate the total cost for the node
                        total_cost = (G.nodes[node]['land_cost'] + 
                                    G.nodes[node]['labor_cost'] * 30 + 
                                    G.nodes[node]['electricity_cost'] * 1000 + 
                                    G.nodes[node]['risk_factor'] * 1000)

                        # Append the node details to the list
                        details.append({
                            "Industrial Area": node,
                            "Latitude": f"{G.nodes[node]['lat']:.4f}",
                            "Longitude": f"{G.nodes[node]['lon']:.4f}",
                            "Total Cost (INR)": f"{total_cost:.2f}",  # Add total cost to the details
                            "Land Cost (INR/sq ft)": f"{G.nodes[node]['land_cost']:.2f}",
                            "Labor Cost (INR/day)": f"{G.nodes[node]['labor_cost']:.2f}",
                            "Electricity Cost (INR/unit)": f"{G.nodes[node]['electricity_cost']:.2f}",
                            "Risk Factor": f"{G.nodes[node]['risk_factor']:.2f}",
                            "Cumulative Demand": f"{G.nodes[node]['demand']:.2f}"
                        })
                    
                    # Convert to DataFrame and display as table
                    df_details = pd.DataFrame(details)
                    st.dataframe(df_details)

                with st.expander("Optimized Distribution Paths", expanded=True):
                    st.markdown("*Computes the most efficient routes to satisfy demand using Ant Colony Optimization.*")
                    aco = SupplyChainACO(n_ants, n_iterations, alpha, beta, rho, q0)
                    aco.initialize(dcs, stores, distances, traffic, risk, emissions, operating_costs, demands, inventories)
                    best_solution, best_cost = aco.run()
                    aco.visualize_solution(best_solution, best_cost)

    with tab5:
        st.write("### Forecast: Analyze & Predict Demand Patterns")
        input_data = generate_seasonal_data(n_periods, n_dcs)

        # Store selector
        selected_store = st.selectbox("Select a DC for forecasting:", list(input_data.keys()))

        # Run forecast button
        with st.spinner("Forecasting... This may take a moment."):
            if st.button("Run Forecast"):
                # Forecast for all stores in parallel
                forecasts, avg_forecasts, _ = forecast_all_stores(input_data, forecast_days=10)

                # Expander for chart and forecast table
                with st.expander("Forecasted Demand", expanded=True):
                    st.subheader(f"Demand History and Forecast for {selected_store}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data
                    historical_days = pd.date_range(end='2023-08-16', periods=len(input_data[selected_store]), freq='D')
                    ax.plot(historical_days, input_data[selected_store], label='Historical Demand')
                    
                    # Plot forecasted data
                    forecast_days = pd.date_range(start='2023-08-17', periods=n_forecast_days, freq='D')
                    ax.plot(forecast_days, forecasts[selected_store], label='Forecasted Demand', color='red')
                    
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Demand')
                    ax.legend()
                    ax.set_title(f'Demand History and Forecast for {selected_store}')
                    plt.xticks(rotation=45)
                    
                    st.pyplot(fig)

                    # Display forecasted demands table
                    st.write(f"###### Forecasted Demand for {selected_store} (Next 10 Days)")
                    forecast_df = pd.DataFrame({
                        'Day': range(1, 11),
                        'Forecasted Demand': forecasts[selected_store]
                    })
                    st.table(forecast_df)

                # Expander for top and bottom performing stores
                with st.expander("DC Rankings", expanded=True):
                    st.write("#### DC Performance Rankings")

                    # Sort stores based on average forecasted demand
                    sorted_stores = sorted(avg_forecasts.items(), key=lambda x: x[1], reverse=True)
                    top_performing = sorted_stores[:3]
                    bottom_performing = sorted_stores[-3:]

                    # Display top 3 performing stores
                    st.write("##### Best Performing DCs (Highest Forecasted Demand)")
                    top_performing_df = pd.DataFrame(top_performing, columns=['Store', 'Avg Forecasted Demand'])
                    st.table(top_performing_df)

                    # Display bottom 3 performing stores
                    st.write("##### Bottom 3 Performing DCs (Lowest Forecasted Demand)")
                    bottom_performing_df = pd.DataFrame(bottom_performing, columns=['Store', 'Avg Forecasted Demand'])
                    st.table(bottom_performing_df)

    st.markdown("---")
    st.markdown("*SCAIO Dashboard: Empowering supply chain decisions with AI*")

if __name__ == "__main__":

    main()