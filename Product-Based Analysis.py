# Importing Libraries
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
warnings.filterwarnings("ignore")

# import dataset and clean for analysis
df = pd.read_csv('Dataset/MIS581_CP_Dataset_Cleaned.csv')
df = df.fillna(0)

#  1 Product-based Hypothesis Testing
df1 = df.iloc[:, 5:30]
df1.info()
df1.describe()

#  1.1 Which consumption category is selling most?
plt.figure()
df1 = df1.sum().sort_values(ascending=True)

df1.plot(kind='barh', figsize=(18, 6))
plt.ticklabel_format(style='plain', axis='x')
plt.xlabel('Sum of Dollars Spent')
plt.ylabel('Consumption Category')
plt.title('Ranking of Selling Performance by Consumption Categories')
plt.show()

# this will identify the top and bottom 5 as the variables used in the correlation clustermap
mycols = df1.nlargest(5) + df1.nsmallest(5)
print(sorted(mycols.index))
corrs = df[mycols.index].select_dtypes(include=np.number).corr(method='kendall')

#  1.2 Do consumer categories have a significant relation with other consumer categories?
sns.clustermap(corrs, cmap="Blues", annot=True)
plt.show()
mask = np.triu(np.ones_like(corrs, dtype=bool))
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corrs, mask=mask, cmap="Blues", vmax=.65, center=0,
            annot=True, square=True, linewidths=0.5, cbar_kws={"shrink": .5})
plt.title('Correlation Cluster Mapping of Consumption Categories')
plt.show()
