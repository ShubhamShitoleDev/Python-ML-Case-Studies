#################################################################################
#
#  Project Name  : Marvellous Customer Segmentation System
#  Description   : Loads the Mall Customers dataset, performs exploratory
#                   data analysis, scales features, uses the Elbow Method
#                   to find the optimal number of clusters, applies K-Means
#                   clustering to group customers by Age, Annual Income, and
#                   Spending Score, then visualizes and interprets the
#                   resulting customer segments.
#  Date          : 23-Aug-2026
#  Author        : Shubham Shitole
#
#################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the Mall Customers CSV and prints an EDA summary
#                 (head, info, stats, missing value counts)
# Return Value :  Loaded DataFrame
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_data(data_path):
    df = pd.read_csv(data_path)

    print("Shape of dataset : ", df.shape)
    print("First few records : ")
    print(df.head())

    print(Border)
    print("Column Info : ")
    df.info()

    print(Border)
    print("Statistical Summary : ")
    print(df.describe())

    print(Border)
    print("Missing Values Per Column : ")
    print(df.isnull().sum())

    return df


#################################################################################
#
# Function Name : DisplayPlots
# Input :         Data
# Description :   Visualizes gender distribution and the raw spread of Age,
#                 Annual Income, and Spending Score before clustering
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def DisplayPlots(Data):
    sns.countplot(x="Gender", data=Data)
    plt.title("Gender Distribution")
    plt.show()

    Data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].hist(figsize=(15, 5), bins=20)
    plt.suptitle("Feature Distributions")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=Data)
    plt.title("Annual Income vs Spending Score (Before Clustering)")
    plt.show()


#################################################################################
#
# Function Name : prepare_cluster_data
# Input :         Data
# Description :   Drops the CustomerID identifier column, encodes Gender as
#                 numeric, and scales Age, Annual Income, and Spending Score
#                 using StandardScaler so no single feature dominates the
#                 distance calculation that K-Means relies on
# Return Value :  cleaned_data (original values, with encoded Gender),
#                 X_scaled (numpy array ready for clustering)
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def prepare_cluster_data(Data):
    cleaned_data = Data.drop(columns=["CustomerID"]).copy()

    cleaned_data["Gender"] = cleaned_data["Gender"].map({"Male": 0, "Female": 1})

    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cleaned_data[features])

    print("Features used for clustering : ", features)
    print("Scaled feature shape : ", X_scaled.shape)

    return cleaned_data, X_scaled


#################################################################################
#
# Function Name : find_optimal_k
# Input :         X_scaled, max_k
# Description :   Runs K-Means for k = 1 to max_k and plots the inertia
#                 (within-cluster sum of squares) for each, so the "elbow"
#                 point — where adding more clusters stops meaningfully
#                 reducing inertia — can be visually identified
# Return Value :  None (displays the elbow plot)
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def find_optimal_k(X_scaled, max_k=10):
    inertia_values = []

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertia_values.append(model.inertia_)
        print(f"k = {k} : inertia = {model.inertia_:.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertia_values, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(range(1, max_k + 1))
    plt.grid()
    plt.show()


#################################################################################
#
# Function Name : build_clusters
# Input :         Data, X_scaled, n_clusters
# Description :   Fits K-Means with the chosen number of clusters and adds
#                 the resulting cluster label back onto the original
#                 (unscaled) DataFrame for interpretation
# Return Value :  DataFrame with a new "Cluster" column
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_clusters(Data, X_scaled, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X_scaled)

    result_data = Data.copy()
    result_data["Cluster"] = cluster_labels

    print(f"Formed {n_clusters} clusters")
    print("Customers per cluster : ")
    print(result_data["Cluster"].value_counts().sort_index())

    return result_data


#################################################################################
#
# Function Name : DisplayClusterPlots
# Input :         Data
# Description :   Visualizes the resulting customer segments on Annual
#                 Income vs Spending Score, colored by cluster
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def DisplayClusterPlots(Data):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="Annual Income (k$)", y="Spending Score (1-100)",
        hue="Cluster", palette="tab10", data=Data
    )
    plt.title("Customer Segments (K-Means Clustering)")
    plt.show()


#################################################################################
#
# Function Name : interpret_clusters
# Input :         Data
# Description :   Prints the average Age, Annual Income, and Spending Score
#                 per cluster, to give each segment a human-readable profile
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def interpret_clusters(Data):
    print(Border)
    print("Cluster Profiles (Average values per segment) : ")
    print(Border)

    profile = Data.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()
    print(profile)


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full Customer Segmentation pipeline end-to-end:
#                 load, visualize, scale, find optimal k, cluster, visualize
#                 segments, and interpret each segment's profile
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    df = load_data("Mall_Customers.csv")

    print(Border)
    print("Step 2 : Visualization of DataSet (Before Clustering)")
    print(Border)

    DisplayPlots(df)

    print(Border)
    print("Step 3 : Prepare Data for Clustering")
    print(Border)

    cleaned_data, X_scaled = prepare_cluster_data(df)

    print(Border)
    print("Step 4 : Find Optimal Number of Clusters (Elbow Method)")
    print(Border)

    find_optimal_k(X_scaled, max_k=10)

    optimal_k = 5

    print(Border)
    print(f"Step 5 : Build Clusters (k = {optimal_k})")
    print(Border)

    clustered_data = build_clusters(cleaned_data, X_scaled, optimal_k)

    print(Border)
    print("Step 6 : Visualize Customer Segments")
    print(Border)

    DisplayClusterPlots(clustered_data)

    print(Border)
    print("Step 7 : Interpret Segments")
    print(Border)

    interpret_clusters(clustered_data)


if __name__ == "__main__":
    main()
