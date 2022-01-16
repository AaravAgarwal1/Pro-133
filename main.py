#This is Pro-132
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df2 = pd.read_csv("star_with_gravity.csv") 
df2

# #Creating into Lists
# mass2 = df2["Mass"].to_list() 

# radius2 = df2["Radius"].to_list() 

# dist2 = df2["Distance"].to_list() 

# gravity2 = df2["Gravity"].to_list() 

# #Sorting Lists

# radius2.sort() 

# mass2.sort() 

# gravity2.sort() 

# dist2.sort() 

#Plotting Graph
# sns.lineplot(mass2,radius2,marker='o',color='red') 
# plt.title("Radius & Mass of the Star") 
# plt.xlabel("Radius") 
# plt.ylabel("Mass") 
# plt.show() #Plotting Figure 2 plt.figure(figsize=(10,5)) sns.lineplot(mass2,gravity2,marker='o',color='red') #planet mass on x axis, gravity is y axis, marker is the dot, and color is red plt.title('Mass and Gravity') plt.xlabel('Mass') #x axis is number of clusters plt.ylabel('Gravity') plt.show()

#**THIS IS PROJECT 133**

from sklearn.cluster import KMeans
import pandas as pd

print(df2)

X = df2.iloc[:,[3,4]].values #getting mass and radius

wcss= [ ] #will hold wcss values
for i in range(1,11): #iterating 10 times, as we expect number of clusters lying between 1 to 11
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42) #making kmeans model, this is method...random state 42 (most of the time this is only used)
  kmeans.fit(X) #fitting the X list in kmeans model

    #intertia mehtod returns wcss for that model
  wcss.append(kmeans.inertia_) #it will give centre value for each cluster in X

plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel('Number of Clusters')
plt.show()