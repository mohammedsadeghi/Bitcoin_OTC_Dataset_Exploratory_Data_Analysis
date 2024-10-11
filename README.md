# Bitcoin OTC Dataset Exploratory Data Analysis (EDA)

This project explores the **Bitcoin OTC** trust network dataset using Python and **Object-Oriented Programming (OOP)** principles. The dataset consists of trust ratings between users in a Bitcoin over-the-counter (OTC) market, and the project performs exploratory data analysis (EDA), network analysis, and visualizations. 

Plots of trust ratings, network statistics, and subgraphs are saved as image files for easy reference.

## Features

- **Object-Oriented Design**: The analysis is encapsulated in a Python class for better structure and reusability.
- **Network Analysis**: The dataset is represented as a directed graph, and various analyses such as degree distribution, centrality, PageRank, and community detection are performed.
- **Plotting and Visualization**: Graphical representations of the network and trust relationships are saved as `.png` files.
- **Modular Methods**: Each step, from loading the data to plotting, is handled by modular methods.

## Dataset

The **Bitcoin OTC dataset** used in this project contains user ratings in a peer-to-peer marketplace. The dataset includes:
- **Source (User A)**: The user giving the rating.
- **Target (User B)**: The user receiving the rating.
- **Rating**: The trust score given by User A to User B.
- **Timestamp**: The time the rating was given (in Unix time).

The dataset is loaded from a CSV file (`soc-sign-bitcoinotc.csv`), which should be placed in the same directory as the Python script.

## Requirements

The project requires the following Python libraries:
```bash
pip install pandas seaborn matplotlib networkx scipy