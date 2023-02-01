import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


directorioclases = "/home/ignacio/thesis/em/maddpg/multiagent-particle-envs/obs no sep/simulated_data/"
df = pd.read_excel(directorioclases+'DataIt08.xlsx')

print(df)

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics

n_clusters = [] 
for current_step in range(36):
    # Create empty list
    S=[]

    # Range of clusters to try (2 to 10)
    K=range(2,4)

    X = df[df["current_step"] == current_step ].loc[:,df.columns!="current_step"]
    for k in K:
        # Set the model and its parameters
        model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans')
    # Fit the model 
        labels = model.fit_predict(X)
    # Calculate Silhoutte Score and append to a list
        S.append(metrics.silhouette_score(X, labels, metric='euclidean'))


    
    n_clusters.append(np.argmax(np.array(S)) + 2)
    
    gm = GaussianMixture(n_components= 3 , covariance_type= "full", random_state=0).fit(X)

    labels = gm.predict(X)
    plt.scatter(X["meantemp"], X["humidity"], c=labels, s=40, cmap='viridis');
    plt.show()
print(n_clusters)
"""
    gm = GaussianMixture(n_components= 2, covariance_type= "full", random_state=0).fit(step_data)

    labels = gm.predict(step_data)
    plt.scatter(step_data["meantemp"], step_data["humidity"], c=labels, s=40, cmap='viridis');
    plt.show()
    # Plot the resulting Silhouette scores on a graph
    plt.figure(figsize=(16,8), dpi=300)
    plt.plot(K, S, 'bo-', color='black')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Identify the number of clusters using Silhouette Score')
    plt.show()
    
    
# Create empty list
S=[]

# Range of clusters to try (2 to 10)
K=range(2,11)

# Select data for clustering model
X = df_loc[['Latitude', 'Longitude']]

for k in K:
    # Set the model and its parameters
    model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans')
    # Fit the model 
    labels = model.fit_predict(X)
    # Calculate Silhoutte Score and append to a list
    S.append(metrics.silhouette_score(X, labels, metric='euclidean'))

# Plot the resulting Silhouette scores on a graph
plt.figure(figsize=(16,8), dpi=300)
plt.plot(K, S, 'bo-', color='black')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Identify the number of clusters using Silhouette Score')
plt.show()
"""
