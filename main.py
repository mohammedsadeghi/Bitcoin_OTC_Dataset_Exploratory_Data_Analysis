import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import os
from datetime import datetime


class BitcoinOTCAnalysis:
    def __init__(self, dataset_path, output_dir='plots'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.df = None
        self.G = None

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """Load the dataset and preprocess it."""
        self.df = pd.read_csv(self.dataset_path, names=['source', 'target', 'rating', 'timestamp'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit='s')  # Convert timestamp to datetime
        print("Data loaded successfully!")
        print(self.df.head())

    def save_plot(self, fig, plot_name):
        """Save a plot to the output directory."""
        plot_path = os.path.join(self.output_dir, plot_name)
        fig.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    def summary_statistics(self):
        """Print summary statistics and save rating distribution plot."""
        print("\nSummary Statistics of Trust Ratings:")
        print(self.df['rating'].describe())

        # Plotting the distribution of trust ratings
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['rating'], bins=20, kde=True, color='skyblue')
        plt.title('Distribution of Trust Ratings')
        plt.xlabel('Trust Rating')
        plt.ylabel('Count')
        fig = plt.gcf()  # Get the current figure
        self.save_plot(fig, 'rating_distribution.png')

    def time_based_analysis(self):
        """Analyze trust ratings over time and save plot."""
        self.df['year'] = self.df['timestamp'].dt.year
        yearly_ratings = self.df.groupby('year')['rating'].mean()

        # Plot average trust ratings by year
        plt.figure(figsize=(10, 6))
        yearly_ratings.plot(kind='bar', color='skyblue')
        plt.title('Average Trust Ratings Per Year')
        plt.xlabel('Year')
        plt.ylabel('Average Rating')
        fig = plt.gcf()
        self.save_plot(fig, 'average_ratings_per_year.png')

    def build_graph(self):
        """Build a directed graph from the dataset."""
        self.G = nx.from_pandas_edgelist(self.df, 'source', 'target', ['rating'], create_using=nx.DiGraph())

        # Manually print the graph information
        print(f"\nGraph Information:")
        print(f"Number of nodes: {self.G.number_of_nodes()}")
        print(f"Number of edges: {self.G.number_of_edges()}")
        print(f"Is the graph directed? {self.G.is_directed()}")

    def degree_analysis(self):
        """Analyze in-degree and out-degree distributions and save the plots."""
        in_degrees = self.G.in_degree()
        out_degrees = self.G.out_degree()

        # Convert to pandas for easier analysis
        in_degree_df = pd.DataFrame(in_degrees, columns=['user', 'in_degree'])
        out_degree_df = pd.DataFrame(out_degrees, columns=['user', 'out_degree'])

        # In-Degree Plot (Trust Received)
        plt.figure(figsize=(10, 6))
        sns.histplot(in_degree_df['in_degree'], bins=50, kde=True, color='green')
        plt.title('In-Degree (Trust Received) Distribution')
        plt.xlabel('In-Degree (Number of Users that Trust)')
        plt.ylabel('Count')
        fig = plt.gcf()
        self.save_plot(fig, 'in_degree_distribution.png')

        # Out-Degree Plot (Trust Given)
        plt.figure(figsize=(10, 6))
        sns.histplot(out_degree_df['out_degree'], bins=50, kde=True, color='orange')
        plt.title('Out-Degree (Trust Given) Distribution')
        plt.xlabel('Out-Degree (Number of Users Trusted)')
        plt.ylabel('Count')
        fig = plt.gcf()
        self.save_plot(fig, 'out_degree_distribution.png')

    def centrality_analysis(self):
        """Calculate and display centrality measures (betweenness centrality and PageRank)."""
        # Betweenness Centrality
        betweenness_centrality = nx.betweenness_centrality(self.G)
        central_users = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 Users by Betweenness Centrality:")
        for user, score in central_users:
            print(f"User {user} - Betweenness Centrality: {score:.6f}")

        # PageRank
        pagerank = nx.pagerank(self.G)
        top_pagerank_users = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 Users by PageRank:")
        for user, score in top_pagerank_users:
            print(f"User {user} - PageRank: {score:.6f}")

    def shortest_path_analysis(self, source_user=1):
        """Perform shortest path analysis for a sample user."""
        print(f"\nShortest path analysis from User {source_user}:")
        try:
            paths = nx.single_source_shortest_path(self.G, source_user)
            for target, path in list(paths.items())[:5]:  # Displaying first 5 paths for brevity
                print(f"Path to User {target}: {path}")
        except nx.NetworkXNoPath:
            print(f"No paths found from User {source_user}")

    def build_graph(self):
        """Build a directed graph from the dataset."""
        self.G = nx.from_pandas_edgelist(self.df, 'source', 'target', ['rating'], create_using=nx.DiGraph())

        # Manually print the graph information
        print(f"\nGraph Information:")
        print(f"Number of nodes: {self.G.number_of_nodes()}")
        print(f"Number of edges: {self.G.number_of_edges()}")
        print(f"Is the graph directed? {self.G.is_directed()}")

    def plot_graph(self, num_nodes=100):
        """
        Plot a subgraph of the full Bitcoin OTC graph.
        :param num_nodes: Number of nodes to include in the subgraph for visualization.
        """
        # Take a subset of the nodes to avoid overloading the plot
        subgraph = self.G.subgraph(list(self.G.nodes())[:num_nodes])

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(subgraph, seed=42)  # spring layout for better visualization

        # Draw the nodes and edges
        nx.draw(subgraph, pos, node_size=50, node_color='skyblue', with_labels=True, font_size=8, edge_color='gray')

        plt.title(f"Subgraph of Bitcoin OTC Trust Network ({num_nodes} nodes)")
        fig = plt.gcf()

        # Save the plot
        self.save_plot(fig, f'subgraph_{num_nodes}_nodes.png')
        plt.show()
    def community_detection(self):
        """Perform community detection and display the first-level communities."""
        from networkx.algorithms.community import girvan_newman
        communities = girvan_newman(self.G)
        first_communities = next(communities)
        sorted_communities = [sorted(list(c)) for c in first_communities]

        print("\nFirst-level communities detected (displaying first 2):")
        for community in sorted_communities[:2]:  # Displaying the first 2 communities
            print(community)


# Example usage
if __name__ == "__main__":
    # Specify the dataset path and output directory for plots
    dataset_path = 'soc-sign-bitcoinotc.csv'

    # Instantiate the class and perform the analysis
    btc_analysis = BitcoinOTCAnalysis(dataset_path)
    btc_analysis.load_data()
    btc_analysis.summary_statistics()
    btc_analysis.time_based_analysis()
    btc_analysis.build_graph()
    btc_analysis.plot_graph(num_nodes=100)  # Plot a subgraph with 100 nodes
    btc_analysis.degree_analysis()
    btc_analysis.centrality_analysis()
    btc_analysis.shortest_path_analysis()
    btc_analysis.community_detection()