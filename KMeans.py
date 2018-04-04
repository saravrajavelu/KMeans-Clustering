
# coding: utf-8

# # KMeans Clustering

# In[1]:


# Loading libraries
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering, DBSCAN, MiniBatchKMeans, Birch
from sklearn.cluster import KMeans as KMeansPack

get_ipython().run_line_magic('matplotlib', 'inline')

def printmd(string):
    display(Markdown(string))


# # 1. Functions

# In[2]:


# Functions of KMeans

def initializeCentroids(k, d, low = 0, high = 1):
    return np.random.uniform(low, high, size = (k,d))

def setClusters(data, centroids, euclidean):
    return np.apply_along_axis(euclidean, 1, data, centroids)


def euclideanRank(centroids, point, nsmall = 0):
    return rankCluster(np.sqrt( np.sum( np.power( np.subtract(centroids, point), 2), axis = 1)), nsmall)

def euclideanDist(centroids, point):
    return np.sqrt( np.sum( np.power( np.subtract(centroids, point), 2), axis = len(centroids.shape)-1)) # Hack

def rankCluster(y, n):
    return np.argpartition(y, n)[n]+1

def updateCentroids(data, clusters, centroids):
    for i in range(centroids.shape[0]):
        if i+1 in clusters:
            centroids[i, ] = data[np.where(clusters == i+1), ].mean(axis = 1)
    return centroids

def nSmallestDistPoints(points, n):
    l = []
    for i in range(points.shape[0]):
        l = np.append(l, euclideanDist(points, points[i, ]))

    n *= 2
    ind = np.argpartition(l, points.shape[0] + n)[points.shape[0] + n]
    r = int(np.floor(ind / points.shape[0]))
    c = int(np.floor(ind % points.shape[0]))
    return r, c, l[ind]

def reallyreallyInitializeCentroids(data, k):
    centroids = data[np.random.randint(0, data.shape[0], k), ]
    replacement_counter = 0
    for i in range(data.shape[0]):
        case = data[i, ]
        closest1, closest2, dist = nSmallestDistPoints(centroids, 1)
        if np.min(euclideanDist(centroids, case)) > dist:
            if euclideanDist(centroids[closest1, ], case) > euclideanDist(centroids[closest2, ], case):
                centroids[closest2, ] = case
            else:
                centroids[closest1, ] = case
            replacement_counter += 1
        else:
            second_closest_case = np.argpartition(euclideanDist(centroids, case), 1)[1]
            first_closest_case = np.argpartition(euclideanDist(centroids, case), 0)[0]
            closest_to_second_closest_case = np.argpartition(euclideanDist(centroids, centroids[second_closest_case, ]), 1)[1]
            #closest1, closest2, dist = nSmallestDistPoints(centroids, 2)

            if np.partition(euclideanDist(centroids, case), 1)[1] > euclideanDist(centroids[second_closest_case, ], centroids[closest_to_second_closest_case, ]):
                centroids[first_closest_case, ] = case
                replacement_counter += 1
    #print('Centroids replaced %d time(s).' % (replacement_counter))
    return centroids

def KMeansFormal(k, data, centroids = None):
    if centroids is None:
        centroids = initializeCentroids(k, data.shape[1], data.min(), data.max())
    p_centroids = centroids - 1
    run_counter = 0
    while not (p_centroids == centroids).all():
        run_counter += 1
        p_centroids = centroids.copy()
        clusters = setClusters(data, centroids, euclideanRank)
        centroids = updateCentroids(data, clusters, centroids)
    
    return pd.DataFrame({'RowID' : [i for i in range(len(clusters))], 'Cluster' : clusters})

def KMeans(k, data, centroids = None):
    if centroids is None:
        centroids = initializeCentroids(k, data.shape[1], data.min(), data.max())
    p_centroids = centroids - 1
    run_counter = 0
    while not (p_centroids == centroids).all():
        run_counter += 1
        p_centroids = centroids.copy()
        clusters = setClusters(data, centroids, euclideanRank)
        centroids = updateCentroids(data, clusters, centroids)
    return centroids, run_counter, clusters

def calculateMeasures(data, clusters, centroids):
    SSE, SSB = 0, 0
    SSE_cluster = []
    for i in range(centroids.shape[0]):
        SSB += euclideanDist(np.mean(data, axis = 0), centroids[i]) * len(np.where(clusters == i+1)[0])
        cluster_filter = np.where(clusters == i+1)
        if len(cluster_filter[0]) == 0:
            temp = 0
        else:
            temp = np.sum(np.apply_along_axis(euclideanDist, 1, data[cluster_filter], centroids[i]))
        SSE += temp
        SSE_cluster.append(temp)
        
    return SSE, SSB, SSE_cluster

def softmax(x):

    x = x - np.max(x, axis = x.ndim - 1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis = x.ndim - 1, keepdims = True)
    
    return x


# # 2. Data Load : TwoDimHard

# In[3]:


TwoDim = pd.read_csv('TwoDimHard.csv')

# Convert to a numpy array
Two = np.array(TwoDim[['X.1','X.2']])

display(TwoDim.head(10))


# In[4]:


TwoDim.info()


# In[5]:


TwoDim.drop(columns = ['ID', 'cluster']).plot(kind = 'box')


# ## Sample Output

# In[6]:


KMeansFormal(4, Two).head()


# ## True Cluster Membership

# In[8]:


groups = TwoDim.groupby('cluster')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# Plot
fig, ax = plt.subplots()
ax.margins(0.05) 
for name, group in groups:
    ax.scatter(group['X.1'], group['X.2'],alpha=0.8, c=colors[name], edgecolors='none', label='Cluster '+str(name))
ax.legend()


#printmd('## True Cluster Membership')
SSE, SSB, SSE_cluster = calculateMeasures(Two, TwoDim.cluster.values, updateCentroids(Two, TwoDim.cluster.values, np.zeros((4,Two.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))
    
plt.show()


# ## K = 4

# In[9]:


# Random Centroids
SSE, SSB = [], []
for i in range(100):
    centroids, iterations, clusters = KMeans(4, Two)
    t1, t2, _ = calculateMeasures(Two, clusters, centroids)
    SSE.append(t1)
    SSB.append(t2)
    
print('Mean SSE : %f & Mean SSB : %f' % (np.mean(SSE), np.mean(SSB)))

x = round(pd.DataFrame({'SSE' : SSE, 'SSB' : SSB}),2)
display(pd.Series(x.SSE.astype(str) + ' | ' + x.SSB.astype(str)).value_counts())


# In[10]:


centroids, iterations, clusters = KMeans(4, Two)
#(TwoDim.cluster == clusters).value_counts()

labels = ['Cluster ' + str(i) for i in np.unique(clusters)]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for data, color, group in zip(np.unique(clusters), colors, labels):
    x = Two[np.where(clusters == data)]
    ax.scatter(x[:, 0], x[:, 1], alpha=0.8, c=color, edgecolors='none', label=group)
plt.legend(loc=1)


printmd('## Clustering Plot k = 4')
SSE, SSB, SSE_cluster = calculateMeasures(Two, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))

print('\nIteration till convergence : %d' %(iterations))
plt.show()


# In[12]:


new_clusters = []
for i in clusters:
    if i == 1:
        new_clusters.append(1)
    elif i == 3:
        new_clusters.append(2)
    elif i == 2:
        new_clusters.append(3)
    elif i == 4:
        new_clusters.append(4)


display(pd.crosstab(pd.Series(new_clusters, name = 'Assigned Cluster'), pd.Series(TwoDim.cluster, name = 'True Clusters')))


# ## K = 3

# In[15]:


# Random Centroids
SSE, SSB = [], []
for i in range(100):
    centroids, iterations, clusters = KMeans(3, Two)
    t1, t2, _ = calculateMeasures(Two, clusters, centroids)
    SSE.append(t1)
    SSB.append(t2)
    
print('Mean SSE : %f & Mean SSB : %f' % (np.mean(SSE), np.mean(SSB)))

x = round(pd.DataFrame({'SSE' : SSE, 'SSB' : SSB}),2)
display(pd.Series(x.SSE.astype(str) + ' | ' + x.SSB.astype(str)).value_counts())


# In[18]:


centroids, iterations, clusters = KMeans(3, Two)
#(TwoDim.cluster == clusters).value_counts()

labels = ['Cluster ' + str(i) for i in np.unique(clusters)]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for data, color, group in zip(np.unique(clusters), colors, labels):
    x = Two[np.where(clusters == data)]
    ax.scatter(x[:, 0], x[:, 1], alpha=0.8, c=color, edgecolors='none', label=group)
plt.legend(loc=1)


printmd('## Clustering Plot k = 3')
SSE, SSB, SSE_cluster = calculateMeasures(Two, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))

print('\nIteration till convergence : %d' %(iterations))
plt.show()


# In[20]:


new_clusters = []
for i in clusters:
    if i == 3:
        new_clusters.append(1)
    elif i == 1:
        new_clusters.append(3)
    elif i == 2:
        new_clusters.append(4)
    elif i == 4:
        new_clusters.append(4)


display(pd.crosstab(pd.Series(new_clusters, name = 'Assigned Cluster'), pd.Series(TwoDim.cluster, name = 'True Clusters')))


# In[21]:


# Sampling Centroids
SSE, SSB = [], []
for i in range(100):
    cent = reallyreallyInitializeCentroids(Two, 4)
    centroids, iterations, clusters = KMeans(4, Two, cent)
    t1, t2, _ = calculateMeasures(Two, clusters, centroids)
    SSE.append(t1)
    SSB.append(t2)
    
print('Mean SSE : %f & Mean SSB : %f' % (np.mean(SSE), np.mean(SSB)))


# In[22]:


cent = reallyreallyInitializeCentroids(Two, 4)
centroids, iterations, clusters = KMeans(4, Two, cent)
#(TwoDim.cluster == clusters).value_counts()

labels = ['Cluster ' + str(i) for i in np.unique(clusters)]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for data, color, group in zip(np.unique(clusters), colors, labels):
    x = Two[np.where(clusters == data)]
    ax.scatter(x[:, 0], x[:, 1], alpha=0.8, c=color, edgecolors='none', label=group)
plt.legend(loc=1)


printmd('## Clustering Plot')
SSE, SSB, SSE_cluster = calculateMeasures(Two, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))


plt.show()


# ## Ideal K Value

# In[24]:


SSE_per_k = []
for j in range(1,11):
    #print('Calculating for k : %d' % (j))
    SSE, SSB = [], []
    for i in range(10):
        centroids, iterations, clusters = KMeans(j, Two)
        t1, t2, _ = calculateMeasures(Two, clusters, centroids)
        SSE.append(t1)
        SSB.append(t2)
    SSE_per_k.append(np.mean(SSE))


# In[25]:


#SSE_per_k
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = pd.Series(SSE_per_k, name = 'SSE').plot(kind = 'line')
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')


# # Dataset 2 : Wine Data Clustering

# In[26]:


# Data Load
wine = pd.read_csv('winequality-red.csv', sep = ';')
Wine_Base = np.array(wine.drop(columns = ['quality','citric acid','density' ,'total sulfur dioxide']))


scaler = preprocessing.MinMaxScaler()
scaler.fit(Wine_Base)
Wine_Norm = scaler.transform(Wine_Base)
Wine_Softmax = softmax(Wine_Base)


# In[27]:


wine.quality.value_counts()


# ## Non normalized data

# In[28]:


pd.DataFrame(Wine_Base).plot(kind = 'box')


# In[32]:


centroids, iterations, clusters = KMeans(10, Wine_Base)
print('Number of iterations : %d' % (iterations))
print(pd.Series(clusters).value_counts())

print('\n')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
#for i,x in enumerate(SSE_cluster):
    #print('Cluster %d: %f' %(i+1, round(x,2)))


# In[33]:


k = 10
cent = reallyreallyInitializeCentroids(Wine_Base, k)
centroids, iterations, clusters = KMeans(k, Wine_Base, cent)
print('Number of iterations : %d' % (iterations))
print(pd.Series(clusters).value_counts())

print('\n')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
#for i,x in enumerate(SSE_cluster):
    #print('Cluster %d: %f' %(i+1, round(x,2)))


# # Sampling with Conditions

# In[35]:


SSE_per_k_Base = []
for j in range(3,20):
    #print('Calculating for k : %d' % (j))
    SSE, SSB = [], []
    for i in range(10):
        cent = reallyreallyInitializeCentroids(Wine_Base, j)
        centroids, iterations, clusters = KMeans(j, Wine_Base, cent)
        t1, t2, _ = calculateMeasures(Wine_Base, clusters, centroids)
        SSE.append(t1)
        SSB.append(t2)
    SSE_per_k_Base.append(np.mean(SSE))


# In[36]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = pd.Series(SSE_per_k_Base, name = 'SSE').plot(kind = 'line')
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')


# ## Random Sampling

# In[38]:


SSE_per_k_Base_Rand = []
for j in range(3,20):
    #print('Calculating for k : %d' % (j))
    SSE, SSB = [], []
    for i in range(10):
        centroids, iterations, clusters = KMeans(j, Wine_Base)
        t1, t2, _ = calculateMeasures(Wine_Base, clusters, centroids)
        SSE.append(t1)
        SSB.append(t2)
    SSE_per_k_Base_Rand.append(np.mean(SSE))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = pd.Series(SSE_per_k_Base_Rand, name = 'SSE').plot(kind = 'line')
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')


# In[39]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = pd.DataFrame({'Sampling with Condition': SSE_per_k_Base, 'Random Initialization' : SSE_per_k_Base_Rand}).plot()
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')


# # Min-Max Normalizaed

# In[40]:


pd.DataFrame(Wine_Norm).plot(kind = 'box')


# In[41]:


centroids, iterations, clusters = KMeans(10, Wine_Norm)
print('Number of iterations : %d' % (iterations))
print(pd.Series(clusters).value_counts())

print('\n')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))


# In[42]:


k = 10
cent = reallyreallyInitializeCentroids(Wine_Norm, k)
centroids, iterations, clusters = KMeans(k, Wine_Norm, cent)
print('Number of iterations : %d' % (iterations))
print(pd.Series(clusters).value_counts())

print('\n')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))


# In[43]:


SSE_per_k_Norm = []
for j in range(3,20):
    #print('Calculating for k : %d' % (j))
    SSE, SSB = [], []
    for i in range(10):
        cent = reallyreallyInitializeCentroids(Wine_Norm, j)
        centroids, iterations, clusters = KMeans(j, Wine_Norm, cent)
        t1, t2, _ = calculateMeasures(Wine_Norm, clusters, centroids)
        SSE.append(t1)
        SSB.append(t2)
    SSE_per_k_Norm.append(np.mean(SSE))


# In[44]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = pd.Series(SSE_per_k_Norm, name = 'SSE').plot(kind = 'line')
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')


# # Softmax Normalizaed

# In[45]:


pd.DataFrame(Wine_Softmax).plot(kind = 'box')


# In[46]:


centroids, iterations, clusters = KMeans(10, Wine_Softmax)
print('Number of iterations : %d' % (iterations))
print(pd.Series(clusters).value_counts())

print('\n')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Softmax, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))


# In[47]:


k = 10
cent = reallyreallyInitializeCentroids(Wine_Softmax, k)
centroids, iterations, clusters = KMeans(k, Wine_Softmax, cent)
print('Number of iterations : %d' % (iterations))
print(pd.Series(clusters).value_counts())

print('\n')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Softmax, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))


# In[49]:


SSE_per_k_Soft = []
for j in range(3,20):
    #print('Calculating for k : %d' % (j))
    SSE, SSB = [], []
    for i in range(10):
        cent = reallyreallyInitializeCentroids(Wine_Softmax, j)
        centroids, iterations, clusters = KMeans(j, Wine_Softmax, cent)
        t1, t2, _ = calculateMeasures(Wine_Softmax, clusters, centroids)
        SSE.append(t1)
        SSB.append(t2)
    SSE_per_k_Soft.append(np.mean(SSE))


# In[50]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax = pd.Series(SSE_per_k_Soft, name = 'SSE').plot(kind = 'line')
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')


# In[51]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
pd.DataFrame({'Non Normalized Data' : SSE_per_k_Base, 'Min-Max Normalized Data': SSE_per_k_Norm, 'Softmax Normalized Data' : SSE_per_k_Soft}).plot()
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')


# In[54]:


get_ipython().run_line_magic('matplotlib', 'notebook')
fig = plt.figure()
ax = pd.DataFrame({'Min-Max Normalized Data': SSE_per_k_Norm, 'Softmax Normalized Data' : SSE_per_k_Soft}).plot()
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')
fig.show()


# In[ ]:


fig, axs = plt.subplots(1,3, figsize=(15, 6))

ax = pd.Series(SSE_per_k_Soft, name = 'SSE').plot(kind = 'line', ax =  axs[2], title =  'Base Data')
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')
ax = pd.Series(SSE_per_k_Norm, name = 'SSE').plot(kind = 'line', ax =  axs[1], title =  'Min-Max Normalized Dataset')
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')
ax = pd.Series(SSE_per_k_Base, name = 'SSE').plot(kind = 'line', ax =  axs[0], title =  'Softmax Normalized Dataset')
ax.set_xlabel('K Value')
ax.set_ylabel('SSE')


# # Comparing with quality 

# In[55]:


printmd('## Quality as Cluster Membership')
k =10

printmd('### Softmax')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Softmax, wine.quality.values, updateCentroids(Wine_Softmax, wine.quality.values, np.zeros((k,Wine_Softmax.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))
    
    
printmd('### Base')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, wine.quality.values, updateCentroids(Wine_Base, wine.quality.values, np.zeros((k,Wine_Base.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))
    
printmd('### Min-Max')
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, wine.quality.values, updateCentroids(Wine_Norm, wine.quality.values, np.zeros((k,Wine_Norm.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))


# In[56]:


k = 6
cent = reallyreallyInitializeCentroids(Wine_Norm, k)
centroids, iterations, clusters = KMeans(k, Wine_Norm, cent)
print('Number of iterations : %d' % (iterations))
print(pd.Series(clusters).value_counts())

SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))


# In[57]:


wine_eda = wine.copy()
wine_eda['cal_qual'] = clusters
wine_eda.drop(columns = ['quality','citric acid','density' ,'total sulfur dioxide'], inplace = True)
wine_eda.groupby('cal_qual').mean()


# In[58]:


wine.columns


# In[59]:


def smaller():
    Wine_Base = np.array(wine.drop(columns = ['quality','citric acid','density' ,'total sulfur dioxide','pH', 'volatile acidity']))
    print(Wine_Base.shape)
    scaler1 = preprocessing.MinMaxScaler()
    scaler1.fit(Wine_Base)
    Wine_Norm = scaler1.transform(Wine_Base)

    k = 6
    cent = reallyreallyInitializeCentroids(Wine_Norm, k)
    centroids, iterations, clusters = KMeans(k, Wine_Norm, cent)
    print('Number of iterations : %d' % (iterations))
    print(pd.Series(clusters).value_counts())

    SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, clusters, centroids)
    print('SSB : %f' % (SSB))
    print('SSE : %f' % (SSE))
    print('Cluster-wise SSE:')
    for i,x in enumerate(SSE_cluster):
        print('Cluster %d: %f' %(i+1, round(x,2)))

    wine_eda = wine.copy()
    wine_eda['Cluster'] = clusters
    wine_eda.drop(columns = ['quality','citric acid','density' ,'total sulfur dioxide'], inplace = True)
    display(wine_eda.groupby('Cluster').mean())

smaller()


# In[60]:


# Wine_Base = np.array(wine.drop(columns = ['quality','citric acid','density' ,'total sulfur dioxide','pH', 'volatile acidity']))
# Wine_Norm = scaler.transform(Wine_Base)

k = 6
cent = reallyreallyInitializeCentroids(Wine_Norm, k)
centroids, iterations, clusters = KMeans(k, Wine_Norm, cent)
print('Number of iterations : %d' % (iterations))
print(pd.Series(clusters).value_counts())

SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, clusters, centroids)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))

wine_eda = wine.copy()
wine_eda['cal_qual'] = clusters
wine_eda.drop(columns = ['quality','citric acid','density' ,'total sulfur dioxide'], inplace = True)
display(wine_eda.groupby('cal_qual').mean())


# In[61]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
labels = ['Cluster ' + str(i) for i in np.unique(clusters)]
for data, color, group in zip(np.unique(clusters), colors, labels):
    print(data, color, group)
    x = wine_eda['fixed acidity'][np.where(clusters == data)[0]]
    y = wine_eda['alcohol'][np.where(clusters == data)[0]]
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', label=group) 
    ax.set_xlabel('K Value')
    ax.set_ylabel('SSE')

plt.show()
#plt.legend(loc=1)


# In[62]:


get_ipython().run_line_magic('matplotlib', 'notebook')

columns_list = []
for i in range(8):
    for j in range(i+1,8):
        #print(i,j)
        columns_list.append((wine_eda.columns[i], wine_eda.columns[j]))

plt.figure(figsize = (20, 30)).subplots_adjust(hspace=1)
counter = 0
for i,j in columns_list:
    counter += 1
    plt.subplot(30,3, counter)
    for data, color, group in zip(np.unique(clusters), colors, labels):
            x = wine_eda[i][np.where(clusters == data)[0]]
            y = wine_eda[j][np.where(clusters == data)[0]]
            plt.scatter(x, y, alpha=0.8, c=color, edgecolors='none', label=group)
            plt.xlabel(i)
            plt.ylabel(j)

        
plt.show()



# In[ ]:


['fixed acidity', 'volatile acidity', 'residual sugar', 'chlorides','free sulfur dioxide', 'pH', 'sulphates', 'alcohol', 'cal_qual']


# In[63]:


get_ipython().run_line_magic('matplotlib', 'notebook')

cols_of_interest = ['fixed acidity','residual sugar', 'chlorides','free sulfur dioxide', 'pH', 'sulphates', 'alcohol']
columns_list = []
for i in range(8):
    for j in range(i+1,len(cols_of_interest)):
        #print(i,j)
        columns_list.append((cols_of_interest[i], cols_of_interest[j]))

#f = plt.figure(figsize = (20, 30)).subplots_adjust(hspace=1)
fig, axes = plt.subplots(7, 3, figsize=(20, 20))
fig.subplots_adjust(hspace=1)
#fig, ax = plt.subplots()
counter = -1
for i,j in columns_list:
    counter += 1
    for data, color, group in zip(np.unique(clusters), colors, labels):
            x = wine_eda[i][np.where(clusters == data)[0]]
            y = wine_eda[j][np.where(clusters == data)[0]]
            axes[int(counter/3),int(counter%3)].scatter(x, y, alpha=0.8, c=color, edgecolors='none', label=group)
            axes[int(counter/3),int(counter%3)].set_xlabel(i)
            axes[int(counter/3),int(counter%3)].set_ylabel(j)
            #ax.plot()


        
plt.show()


# ## Off the Shelf

# In[64]:



KMS = KMeansPack(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(Two)
print(KMS)
SSE, SSB, SSE_cluster = calculateMeasures(Two, KMS.labels_, KMS.cluster_centers_)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))
print('Cluster-wise SSE:')
for i,x in enumerate(SSE_cluster):
    print('Cluster %d: %f' %(i+1, round(x,2)))


# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for data, color, group in zip(np.unique(KMS.labels_), colors, labels):
    print(data, color)
    x = Two[np.where(KMS.labels_ == data)]
    ax.scatter(x[:, 0], x[:, 1], alpha=0.8, c=color, edgecolors='none', label=group)
plt.show()

new_clusters = []
for i in KMS.labels_:
    if i == 3:
        new_clusters.append(1)
    elif i == 1:
        new_clusters.append(2)
    elif i == 0:
        new_clusters.append(3)
    elif i == 2:
        new_clusters.append(4)


display(pd.crosstab(pd.Series(new_clusters, name = 'Assigned Cluster'), pd.Series(TwoDim.cluster, name = 'True Clusters')))


# In[66]:



print('\nWine_Base')
KMS = KMeansPack(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(Wine_Base)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, KMS.labels_, KMS.cluster_centers_)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

print('\nWine_Norm')
KMS = KMeansPack(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(Wine_Norm)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, KMS.labels_, KMS.cluster_centers_)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

print('\nWine_Softmax')
KMS = KMeansPack(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(Wine_Softmax)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Softmax, KMS.labels_, KMS.cluster_centers_)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

KMS


# In[67]:


k = 6
print('\nWine_Base')
KMS = AgglomerativeClustering(n_clusters=k).fit(Wine_Base)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, KMS.labels_, updateCentroids(Wine_Base, KMS.labels_, np.zeros((k,Wine_Base.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

print('\nWine_Norm')
KMS = AgglomerativeClustering(n_clusters=k).fit(Wine_Norm)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, KMS.labels_, updateCentroids(Wine_Norm, KMS.labels_, np.zeros((k,Wine_Norm.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

print('\nWine_Softmax')
KMS = AgglomerativeClustering(n_clusters=k).fit(Wine_Softmax)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Softmax, KMS.labels_, updateCentroids(Wine_Softmax, KMS.labels_, np.zeros((k,Wine_Softmax.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

KMS


# In[68]:



print('\nWine_Base')
KMS = MiniBatchKMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(Wine_Base)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, KMS.labels_, KMS.cluster_centers_)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

print('\nWine_Norm')
KMS = MiniBatchKMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(Wine_Norm)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Norm, KMS.labels_, KMS.cluster_centers_)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

print('\nWine_Softmax')
KMS = MiniBatchKMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=0.0001).fit(Wine_Softmax)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Softmax, KMS.labels_, KMS.cluster_centers_)
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

KMS


# In[69]:



print('\nWine_Base')
KMS = Birch(n_clusters=6).fit(Wine_Base)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, KMS.labels_, updateCentroids(Wine_Base, KMS.labels_, np.zeros((k,Wine_Base.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

print('\nWine_Norm')
KMS = Birch(n_clusters=6).fit(Wine_Norm)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, KMS.labels_, updateCentroids(Wine_Base, KMS.labels_, np.zeros((k,Wine_Base.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

print('\nWine_Softmax')
KMS = Birch(n_clusters=6).fit(Wine_Softmax)
SSE, SSB, SSE_cluster = calculateMeasures(Wine_Base, KMS.labels_, updateCentroids(Wine_Base, KMS.labels_, np.zeros((k,Wine_Base.shape[1]))))
print('SSB : %f' % (SSB))
print('SSE : %f' % (SSE))

KMS

