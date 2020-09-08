"""
This file attempts to predict who the next Ronaldo, Messi, and Neymar
will be by analyzing data about players under 20 years old from Football
Manager. The K-means cluster technique is used to find which young players
are most similar to the stars of today's game. 
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

filename = 'datasets/football_manager_dataset.csv'
players = pd.read_csv(filename)

ronaldo = players[players['Name'] == 'Cristiano Ronaldo']
messi = players[players['Name'] == 'Lionel Messi']
neymar = players[players['Name'] == 'Neymar']


candidates = players[players['Age'] <= 20]
candidates.append(ronaldo)
candidates.append(messi)
candidates.append(neymar)
features = candidates.iloc[:,22:-22] # removing goalkeeper features and other unimportant features
scaled_features=(features-features.min())/(features.max()-features.min())

kmeans = KMeans(init="random", n_clusters=7, n_init=50, max_iter=500)

kmeans.fit(scaled_features)

ronaldo_cluster = kmeans.labels_[-3]
messi_cluster = kmeans.labels_[-2]
neymar_cluster = kmeans.labels_[-1]


candidates = candidates.assign(cluster=pd.Series(kmeans.labels_).values)

# here we have all the candidates who are most similar to Ronaldo, Messi, and Neymar
next_ronaldos = candidates[candidates['cluster'] == ronaldo_cluster]
next_messis = candidates[candidates['cluster'] == messi_cluster]
next_neymars = candidates[candidates['cluster'] == neymar_cluster]

# seeing the average traits players in the same cluster as each player have
ronaldo_means = next_ronaldos.mean()
messi_means = next_messis.mean()
neymar_means = next_neymars.mean()

means = pd.concat([ronaldo.mean(), ronaldo_means, messi.mean(), messi_means, neymar.mean(), neymar_means], axis=1).transpose()
indices = [means.index[i] for i in range(6)]
means = means.rename(index={indices[0]: 'Ronaldo'})
means = means.rename(index={indices[1]: 'Next Ronaldos'})
means = means.rename(index={indices[2]: 'Messi'})
means = means.rename(index={indices[3]: 'Next Messis'})
means = means.rename(index={indices[4]: 'Neymar'})
means = means.rename(index={indices[5]: 'Next Neymars'})


means.to_csv('datasets/next_great_players_averages.csv')



# The code below was used to determine optimal number of clusters using
# the elbow method and the silhouette coefficient. 
"""
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

silhouette_coefficients = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
"""