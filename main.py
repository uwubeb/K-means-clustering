from collections import Counter

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    SAMPLE_SIZE = 7413
    VAR1 = 'Vitamin B3'
    VAR2 = 'Vitamin B5'
    dataset, classes = make_blobs(n_samples=SAMPLE_SIZE, n_features=2, centers=3, cluster_std=0.5, random_state=0)
    df = pd.DataFrame(dataset, columns=[VAR1, VAR2])
    print(df.head(2))

    kmeans = KMeans(n_clusters=8, init='k-means++', random_state=0).fit(df)
    print(Counter(kmeans.labels_))
    sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
