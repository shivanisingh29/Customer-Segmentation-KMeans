from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("Mall_Customers.csv")
print(df.head())

X=df[['Annual Income (k$)','Spending Score (1-100)']]

model=KMeans(n_clusters=3,random_state=42)

df['Cluster']=model.fit_predict(X)
print(df.head())

plt.scatter(X['Spending Score (1-100)'],X['Annual Income (k$)'],c=df['Cluster'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title("Cluster Segmentation")
plt.show()