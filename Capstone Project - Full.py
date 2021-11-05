# Importing Libraries
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
sns.set_style('darkgrid')

# Import dataset
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

# 2 Consumer-based Hypothesis Testing
# 2.1 Are there existing consumer groups?
# 2.1.1 K-means cluster model

kmeans = KMeans(n_clusters=4)
kmeans.fit(df[['Education', 'Family Type', 'Income Class', 'Division']])
pred = kmeans.predict(df[['Education', 'Family Type', 'Income Class', 'Division']])
print(kmeans.cluster_centers_)
print(kmeans.labels_)

pred = pd.DataFrame(pred, columns=['pred'])
df = df.join(pred)
print()

# 2.1.2 Determine Number of Clusters using Elbow Method
error_rate = []
K = range(1, 16)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['Education', 'Family Type', 'Income Class', 'Division']])
    kmeans.predict(df[['Education', 'Family Type', 'Income Class', 'Division']])
    error_rate.append(kmeans.inertia_)
plt.scatter(K, error_rate, c='red')
plt.plot(K, error_rate, c='blue')
plt.xlabel('clusters')
plt.ylabel('Error Rate')
plt.title('Elbow Method showing optimal # Clusters')
plt.show()
print(error_rate)

# 2.1.3 Create Clusters
cluster = KMeans(n_clusters=4, init='k-means++', random_state=0)
cluster.fit(df[['Education', 'Family Type', 'Income Class', 'Division']])

y_pred = cluster.predict(df[['Education', 'Family Type', 'Income Class', 'Division']])
print(y_pred)

y_col = y_pred.reshape(len(y_pred), 1)
output = np.append(arr=df, values=y_col, axis=1).nonzero()
df.head()
print(output)

# 2.1.4 Visualize Clusters
# 2.1.4.1 Number of Consumers Per Cluster
sns.countplot(x='pred', data=df, palette='magma')
plt.title('Count of Consumers per Cluster')
plt.ylabel('# of Consumers')
plt.xlabel('Clusters')
plt.show()

# 2.1.4.2 as box plots
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 6))
ty = sns.boxplot(x='pred', y='Income Class', data=df, ax=ax[0], palette='magma')
sns.despine(left=True)
ty.set_title('Clusters based on Income Class')
ty.set_ylabel('Income Class')
ty.set_xlabel('Clusters')

tt = sns.boxplot(x='pred', y='Family Type', data=df, ax=ax[1], palette='magma')
tt.set_title('Clusters based on Family Type')
tt.set_ylabel('Family Type')
tt.set_xlabel('Clusters')

tf = sns.boxplot(x='pred', y='Education', data=df, ax=ax[2], palette='magma')
tf.set_title('Clusters based on Education')
tf.set_ylabel('Education')
tf.set_xlabel('Clusters')

tr = sns.boxplot(x='pred', y='Division', data=df, ax=ax[3], palette='magma')
tr.set_title('Clusters based on Regional Division')
tr.set_ylabel('Division')
tr.set_xlabel('Clusters')
plt.show()

# 2.1.4.3 as scatter plots
plt.figure()
df = df.loc[df['Education'] != 0]
cols_to_plot = df.columns[1:5].tolist() + ['pred']

sns.pairplot(data=df[cols_to_plot], hue='pred', diag_kind='kde', palette='magma')
plt.show()

# 3 Location Hypothesis Testing
# 3.1 Is there a relationship between the location of a customer and the products they purchase?
# 3.1.1 Calculate expenditures in each column as Total in each category
Food = [col for col in df.columns if 'Food' in col]
df['Tot_Food'] = df[Food].sum(axis=1)
Transportation = [col for col in df.columns if 'Transportation' in col]
df['Tot_Transportation'] = df[Transportation].sum(axis=1)
Rented_Dwelling = [col for col in df.columns if 'Rented Dwelling' in col]
df['Tot_Rented_Dwelling'] = df[Rented_Dwelling].sum(axis=1)
Eat_Out = [col for col in df.columns if 'Eat Out' in col]
df['Tot_Eat_Out'] = df[Eat_Out].sum(axis=1)
Alcohol_Beverages = [col for col in df.columns if 'Alcohol Beverages' in col]
df['Tot_Alcohol_Beverages'] = df[Alcohol_Beverages].sum(axis=1)
Fuel = [col for col in df.columns if 'Fuel' in col]
df['Tot_Fuel'] = df[Fuel].sum(axis=1)
Babysitting_and_Childcare = [col for col in df.columns if 'Babysitting and Childcare' in col]
df['Tot_Babysitting_and_Childcare'] = df[Babysitting_and_Childcare].sum(axis=1)
Furniture = [col for col in df.columns if 'Furniture' in col]
df['Tot_Furniture'] = df[Furniture].sum(axis=1)
Apparel = [col for col in df.columns if 'Apparel' in col]
df['Tot_Apparel'] = df[Apparel].sum(axis=1)
Male_Clothing = [col for col in df.columns if 'Male Clothing' in col]
df['Tot_Male_Clothing'] = df[Male_Clothing].sum(axis=1)
Men_Clothing = [col for col in df.columns if 'Men Clothing' in col]
df['Tot_Men_Clothing'] = df[Men_Clothing].sum(axis=1)
Footwear = [col for col in df.columns if 'Footwear' in col]
df['Tot_Footwear'] = df[Footwear].sum(axis=1)
Other_Apparel = [col for col in df.columns if 'Other Apparel' in col]
df['Tot_Other_Apparel'] = df[Other_Apparel].sum(axis=1)
Cars_and_Trucks = [col for col in df.columns if 'Cars and Trucks' in col]
df['Tot_Cars_and_Trucks'] = df[Cars_and_Trucks].sum(axis=1)
Gas = [col for col in df.columns if 'Gas' in col]
df['Tot_Gas'] = df[Gas].sum(axis=1)
Public_Transportation = [col for col in df.columns if 'Public Transportation' in col]
df['Tot_Public_Transportation'] = df[Public_Transportation].sum(axis=1)
Pet_and_Toy = [col for col in df.columns if 'Pet and Toy' in col]
df['Tot_Pet_and_Toy'] = df[Pet_and_Toy].sum(axis=1)
Personal_Care = [col for col in df.columns if 'Personal Care' in col]
df['Tot_Personal_Care'] = df[Personal_Care].sum(axis=1)
Household_Textiles = [col for col in df.columns if 'Household Textiles' in col]
df['Tot_Household_Textiles'] = df[Household_Textiles].sum(axis=1)
Reading = [col for col in df.columns if 'Reading' in col]
df['Tot_Reading'] = df[Reading].sum(axis=1)
Boy_Clothing = [col for col in df.columns if 'Boy Clothing' in col]
df['Tot_Boy_Clothing'] = df[Boy_Clothing].sum(axis=1)
Girl_Clothing = [col for col in df.columns if 'Girl Clothing' in col]
df['Tot_Girl_Clothing'] = df[Girl_Clothing].sum(axis=1)
Child_Clothing = [col for col in df.columns if 'Child Clothing' in col]
df['Tot_Child_Clothing'] = df[Child_Clothing].sum(axis=1)
Entertainment = [col for col in df.columns if 'Entertainment' in col]
df['Tot_Entertainment'] = df[Entertainment].sum(axis=1)

# 3.1.2 Plot Matrix of total spending in each category based on defined Cluster Type
fig2, ax = plt.subplots(nrows=4, ncols=6, figsize=(16, 16))
aa = sns.barplot(x='pred', y="Tot_Food", ax=ax[0, 0], data=df, palette='magma')
aa.set_title('Food')
aa.set_ylabel('$ Spent')
aa.set_xlabel('Clusters')
ab = sns.barplot(x='pred', y="Tot_Transportation", ax=ax[0, 1], data=df, palette='magma')
ab.set_title('Transportation')
ab.set_xlabel('Clusters')
ac = sns.barplot(x='pred', y="Tot_Rented_Dwelling", ax=ax[0, 2], data=df, palette='magma')
ac.set_title('Rented Dwelling')
ac.set_xlabel('Clusters')
ad = sns.barplot(x='pred', y="Tot_Entertainment", ax=ax[0, 3], data=df, palette='magma')
ad.set_title('Entertainment')
ad.set_xlabel('Clusters')
ae = sns.barplot(x='pred', y="Tot_Eat_Out", ax=ax[0, 4], data=df, palette='magma')
ae.set_title('Eating Out')
ae.set_xlabel('Clusters')
ag = sns.barplot(x='pred', y="Tot_Fuel", ax=ax[0, 5], data=df, palette='magma')
ag.set_title('Fuel')
ag.set_xlabel('Clusters')
ah = sns.barplot(x='pred', y="Tot_Babysitting_and_Childcare", ax=ax[1, 0], data=df, palette='magma')
ah.set_title('Babysitting and Childcare')
ah.set_ylabel('$ Spent')
ah.set_xlabel('Clusters')
ai = sns.barplot(x='pred', y="Tot_Furniture", ax=ax[1, 1], data=df, palette='magma')
ai.set_title('Furniture')
ai.set_xlabel('Clusters')
aj = sns.barplot(x='pred', y="Tot_Apparel", ax=ax[1, 2], data=df, palette='magma')
aj.set_title('Apparel')
aj.set_xlabel('Clusters')
ak = sns.barplot(x='pred', y="Tot_Male_Clothing", ax=ax[1, 3], data=df, palette='magma')
ak.set_title('Male Clothing')
ak.set_xlabel('Clusters')
al = sns.barplot(x='pred', y="Tot_Men_Clothing", ax=ax[1, 4], data=df, palette='magma')
al.set_title('Men Clothing')
al.set_xlabel('Clusters')
am = sns.barplot(x='pred', y="Tot_Footwear", ax=ax[1, 5], data=df, palette='magma')
am.set_title('Footwear')
am.set_xlabel('Clusters')
an = sns.barplot(x='pred', y="Tot_Other_Apparel", ax=ax[2, 0], data=df, palette='magma')
an.set_title('Other Apparel')
an.set_ylabel('$ Spent')
an.set_xlabel('Clusters')
ao = sns.barplot(x='pred', y="Tot_Cars_and_Trucks", ax=ax[2, 1], data=df, palette='magma')
ao.set_title('Cars and Trucks')
ao.set_xlabel('Clusters')
ap = sns.barplot(x='pred', y="Tot_Gas", ax=ax[2, 2], data=df, palette='magma')
ap.set_title('Gas')
ap.set_xlabel('Clusters')
aq = sns.barplot(x='pred', y="Tot_Public_Transportation", ax=ax[2, 3], data=df, palette='magma')
aq.set_title('Public Transportation')
aq.set_ylabel('$ Spent')
aq.set_xlabel('Clusters')
ar = sns.barplot(x='pred', y="Tot_Pet_and_Toy", ax=ax[2, 4], data=df, palette='magma')
ar.set_title('Pet and Toy')
ar.set_xlabel('Clusters')
at = sns.barplot(x='pred', y="Tot_Personal_Care", ax=ax[2, 5], data=df, palette='magma')
at.set_title('Personal Care')
at.set_xlabel('Clusters')
af = sns.barplot(x='pred', y="Tot_Alcohol_Beverages", ax=ax[3, 0], data=df, palette='magma')
af.set_title('Alcohol Beverages')
af.set_ylabel('$ Spent')
af.set_xlabel('Clusters')
au = sns.barplot(x='pred', y="Tot_Household_Textiles", ax=ax[3, 1], data=df, palette='magma')
au.set_title('Household Textiles')
au.set_xlabel('Clusters')
av = sns.barplot(x='pred', y="Tot_Reading", ax=ax[3, 2], data=df, palette='magma')
av.set_title('Reading')
av.set_xlabel('Clusters')
aw = sns.barplot(x='pred', y="Tot_Boy_Clothing", ax=ax[3, 3], data=df, palette='magma')
aw.set_title('Boy Clothing')
aw.set_xlabel('Clusters')
ay = sns.barplot(x='pred', y="Tot_Girl_Clothing", ax=ax[3, 4], data=df, palette='magma')
ay.set_title('Girl Clothing')
ay.set_xlabel('Clusters')
az = sns.barplot(x='pred', y="Tot_Child_Clothing", ax=ax[3, 5], data=df, palette='magma')
az.set_title('Child Clothing')
az.set_xlabel('Clusters')
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

# 3.1.3 Calculate the mean values for each Consumption Category grouped by location (Division)
meanDivTransportation = df.groupby(by=['Division'])['Transportation'].mean()
print(meanDivTransportation)
meanDivFood = df.groupby(by=['Division'])['Food'].mean()
print(meanDivFood)
meanDivRented_Dwelling = df.groupby(by=['Division'])['Rented Dwelling'].mean()
print(meanDivRented_Dwelling)
meanDivEat_Out = df.groupby(by=['Division'])['Eat Out'].mean()
print(meanDivEat_Out)
meanDivAlcohol_Beverages = df.groupby(by=['Division'])['Alcohol Beverages'].mean()
print(meanDivAlcohol_Beverages)
meanDivFuel = df.groupby(by=['Division'])['Fuel'].mean()
print(meanDivFuel)
meanDivBabysitting_and_Childcare = df.groupby(by=['Division'])['Babysitting and Childcare'].mean()
print(meanDivBabysitting_and_Childcare)
meanDivFurniture = df.groupby(by=['Division'])['Furniture'].mean()
print(meanDivFurniture)
meanDivApparel = df.groupby(by=['Division'])['Apparel'].mean()
print(meanDivApparel)
meanDivMale_Clothing = df.groupby(by=['Division'])['Male Clothing'].mean()
print(meanDivMale_Clothing)
meanDivMen_Clothing = df.groupby(by=['Division'])['Men Clothing'].mean()
print(meanDivMen_Clothing)
meanDivFootwear = df.groupby(by=['Division'])['Footwear'].mean()
print(meanDivFootwear)
meanDivOther_Apparel = df.groupby(by=['Division'])['Other Apparel'].mean()
print(meanDivOther_Apparel)
meanDivCars_and_Trucks = df.groupby(by=['Division'])['Cars and Trucks'].mean()
print(meanDivCars_and_Trucks)
meanDivGas = df.groupby(by=['Division'])['Gas'].mean()
print(meanDivGas)
meanDivPublic_Transportation = df.groupby(by=['Division'])['Public Transportation'].mean()
print(meanDivPublic_Transportation)
meanDivPet_and_Toy = df.groupby(by=['Division'])['Pet and Toy'].mean()
print(meanDivPet_and_Toy)
meanDivPersonal_Care = df.groupby(by=['Division'])['Personal Care'].mean()
print(meanDivPersonal_Care)
meanDivHousehold_Textiles = df.groupby(by=['Division'])['Household Textiles'].mean()
print(meanDivHousehold_Textiles)
meanDivReading = df.groupby(by=['Division'])['Reading'].mean()
print(meanDivReading)
meanDivBoy_Clothing = df.groupby(by=['Division'])['Boy Clothing'].mean()
print(meanDivBoy_Clothing)
meanDivGirl_Clothing = df.groupby(by=['Division'])['Girl Clothing'].mean()
print(meanDivGirl_Clothing)
meanDivChild_Clothing = df.groupby(by=['Division'])['Child Clothing'].mean()
print(meanDivChild_Clothing)
meanDivEntertainment = df.groupby(by=['Division'])['Entertainment'].mean()
print(meanDivEntertainment)

# 3.1.4 Plot Matrix of total spending in each category based location (Division)
fig3, ax = plt.subplots(nrows=4, ncols=6, figsize=(16, 16))
plt.title('Expenditures by Category and Area')
aa = sns.barplot(x='Division', y="Tot_Food", ax=ax[0, 0], data=df, palette='magma')
aa.set_title('Food')
aa.set_ylabel('$ Spent')
aa.set_xlabel('Division')
ab = sns.barplot(x='Division', y="Tot_Transportation", ax=ax[0, 1], data=df, palette='magma')
ab.set_title('Transportation')
ab.set_xlabel('Division')
ac = sns.barplot(x='Division', y="Tot_Rented_Dwelling", ax=ax[0, 2], data=df, palette='magma')
ac.set_title('Rented Dwelling')
ac.set_xlabel('Division')
ad = sns.barplot(x='Division', y="Tot_Entertainment", ax=ax[0, 3], data=df, palette='magma')
ad.set_title('Entertainment')
ad.set_xlabel('Division')
ae = sns.barplot(x='Division', y="Tot_Eat_Out", ax=ax[0, 4], data=df, palette='magma')
ae.set_title('Eating Out')
ae.set_xlabel('Division')
ag = sns.barplot(x='Division', y="Tot_Fuel", ax=ax[0, 5], data=df, palette='magma')
ag.set_title('Fuel')
ag.set_xlabel('Division')
ah = sns.barplot(x='Division', y="Tot_Babysitting_and_Childcare", ax=ax[1, 0], data=df, palette='magma')
ah.set_title('Babysitting and Childcare')
ah.set_ylabel('$ Spent')
ah.set_xlabel('Division')
ai = sns.barplot(x='Division', y="Tot_Furniture", ax=ax[1, 1], data=df, palette='magma')
ai.set_title('Furniture')
ai.set_xlabel('Division')
aj = sns.barplot(x='Division', y="Tot_Apparel", ax=ax[1, 2], data=df, palette='magma')
aj.set_title('Apparel')
aj.set_xlabel('Division')
ak = sns.barplot(x='Division', y="Tot_Male_Clothing", ax=ax[1, 3], data=df, palette='magma')
ak.set_title('Male Clothing')
ak.set_xlabel('Division')
al = sns.barplot(x='Division', y="Tot_Men_Clothing", ax=ax[1, 4], data=df, palette='magma')
al.set_title('Men Clothing')
al.set_xlabel('Division')
am = sns.barplot(x='Division', y="Tot_Footwear", ax=ax[1, 5], data=df, palette='magma')
am.set_title('Footwear')
am.set_xlabel('Division')
an = sns.barplot(x='Division', y="Tot_Other_Apparel", ax=ax[2, 0], data=df, palette='magma')
an.set_title('Other Apparel')
an.set_ylabel('$ Spent')
an.set_xlabel('Division')
ao = sns.barplot(x='Division', y="Tot_Cars_and_Trucks", ax=ax[2, 1], data=df, palette='magma')
ao.set_title('Cars and Trucks')
ao.set_xlabel('Division')
ap = sns.barplot(x='Division', y="Tot_Gas", ax=ax[2, 2], data=df, palette='magma')
ap.set_title('Gas')
ap.set_xlabel('Division')
aq = sns.barplot(x='Division', y="Tot_Public_Transportation", ax=ax[2, 3], data=df, palette='magma')
aq.set_title('Public Transportation')
aq.set_ylabel('$ Spent')
aq.set_xlabel('Division')
ar = sns.barplot(x='Division', y="Tot_Pet_and_Toy", ax=ax[2, 4], data=df, palette='magma')
ar.set_title('Pet and Toy')
ar.set_xlabel('Division')
at = sns.barplot(x='Division', y="Tot_Personal_Care", ax=ax[2, 5], data=df, palette='magma')
at.set_title('Personal Care')
at.set_xlabel('Division')
af = sns.barplot(x='Division', y="Tot_Alcohol_Beverages", ax=ax[3, 0], data=df, palette='magma')
af.set_title('Alcohol Beverages')
af.set_ylabel('$ Spent')
af.set_xlabel('Division')
au = sns.barplot(x='Division', y="Tot_Household_Textiles", ax=ax[3, 1], data=df, palette='magma')
au.set_title('Household Textiles')
au.set_xlabel('Division')
av = sns.barplot(x='Division', y="Tot_Reading", ax=ax[3, 2], data=df, palette='magma')
av.set_title('Reading')
av.set_xlabel('Division')
aw = sns.barplot(x='Division', y="Tot_Boy_Clothing", ax=ax[3, 3], data=df, palette='magma')
aw.set_title('Boy Clothing')
aw.set_xlabel('Division')
ay = sns.barplot(x='Division', y="Tot_Girl_Clothing", ax=ax[3, 4], data=df, palette='magma')
ay.set_title('Girl Clothing')
ay.set_xlabel('Division')
az = sns.barplot(x='Division', y="Tot_Child_Clothing", ax=ax[3, 5], data=df, palette='magma')
az.set_title('Child Clothing')
az.set_xlabel('Division')

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()
