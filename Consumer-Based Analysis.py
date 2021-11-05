# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')

# import dataset and clean for analysis
df = pd.read_csv('Dataset/MIS581_CP_Dataset_Cleaned.csv')
df = df.fillna(0)

df.head()
df.info()
df.describe()


df_new = df.iloc[:, 1:5]

# kmeans cluster model
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_new)
pred = kmeans.predict(df_new)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

pred = pd.DataFrame(pred, columns=['pred'])
df_new = df_new.join(pred)
print()

# Determine Number of Clusters using Elbow Method
error_rate = []
K = range(1, 16)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_new)
    kmeans.predict(df_new)
    error_rate.append(kmeans.inertia_)
plt.scatter(K, error_rate, c='red')
plt.plot(K, error_rate, c='blue')
plt.xlabel('clusters')
plt.ylabel('Error Rate')
plt.title('Elbow Method showing optimal # Clusters')
plt.show()
print(error_rate)

# Create Clusters
cluster = KMeans(n_clusters=4, init='k-means++', random_state=0)
cluster.fit(df_new)

y_pred = cluster.predict(df_new)
print(y_pred)

y_col = y_pred.reshape(len(y_pred), 1)
output = np.append(arr=df_new, values=y_col, axis=1).nonzero()
df_new.head()
print(output)

# Visualize Clusters
# Number of Consumers Per Cluster
sns.countplot(x='pred', data=df_new, palette='magma')
plt.title('Count of Consumers per Cluster')
plt.ylabel('# of Consumers')
plt.xlabel('Clusters')
plt.show()

# as box plots
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 6))
ty = sns.boxplot(x='pred', y='Income Class', data=df_new, ax=ax[0], palette='magma')
sns.despine(left=True)
ty.set_title('Clusters based on Income Class')
ty.set_ylabel('Income Class')
ty.set_xlabel('Clusters')

tt = sns.boxplot(x='pred', y='Family Type', data=df_new, ax=ax[1], palette='magma')
tt.set_title('Clusters based on Family Type')
tt.set_ylabel('Family Type')
tt.set_xlabel('Clusters')

tf = sns.boxplot(x='pred', y='Education', data=df_new, ax=ax[2], palette='magma')
tf.set_title('Clusters based on Education')
tf.set_ylabel('Education')
tf.set_xlabel('Clusters')

tr = sns.boxplot(x='pred', y='Division', data=df_new, ax=ax[3], palette='magma')
tr.set_title('Clusters based on Regional Division')
tr.set_ylabel('Division')
tr.set_xlabel('Clusters')
plt.show()

# as scatter plots
df_new = df_new.loc[df_new['Education'] != 0]
sns.pairplot(hue='pred', data=df_new, diag_kind='kde', palette='magma')
plt.show()
