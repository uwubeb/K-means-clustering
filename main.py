from collections import Counter

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    SAMPLE_SIZE = 7413
    CLUSTER_COUNT = 5
    VAR1 = 'Saturated Fat'
    VAR2 = 'Vitamin D'
    dataset, classes = make_blobs(n_samples=SAMPLE_SIZE, n_features=2, centers=CLUSTER_COUNT, cluster_std=0.3, random_state=0)
    df = pd.DataFrame(dataset, columns=[VAR1, VAR2])

    print("Variables: ", VAR1, VAR2)
    print("Cluster count: ", CLUSTER_COUNT)
    kmeans = KMeans(n_clusters=CLUSTER_COUNT, init='k-means++', random_state=0).fit(df)
    print(Counter(kmeans.labels_))
    print("Inertia: ", kmeans.inertia_)
    sns.scatterplot(data=df, x=VAR1, y=VAR2, hue=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker="X", c="r", s=80, label="centroids")
    plt.legend()
    plt.show()

    silhouette_avg = silhouette_score(df, kmeans.labels_)
    print("The average silhouette_score is :", silhouette_avg)