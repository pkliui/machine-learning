```python
import numpy as np
from matplotlib import pyplot as plt
import sys

sourcepath = '/Users/Pavel/Documents/repos/machine-learning/6.86x-ml-python/project4/netflix'
sys.path.insert(0,sourcepath)

#automatically track changes in the source code
%load_ext autoreload
%autoreload 2
```

# Project 4 of MITx 6.86x class on Machine Learning in Python


### Useful resources

Higher School of Economics ML course: 
* http://wiki.cs.hse.ru/Машинное_обучение_1
* http://wiki.cs.hse.ru/Машинное_обучение_2
* k-means https://github.com/esokolov/ml-course-hse/blob/master/2020-fall/lecture-notes/lecture11-unsupervised.pdf
* EM-algorithm, lecture: https://github.com/esokolov/ml-course-hse/blob/master/2020-spring/lecture-notes/lecture15-em.pdf
* EM-algorithm, derivation of the log-likelihood: https://github.com/esokolov/ml-course-hse/blob/master/2020-spring/seminars/sem15-em.pdf

# K-means algorithm

* k-means algorithm seeks to partition $l$ observations into $k$ clusters in which each observation belongs to the cluster with the nearest mean (clusters center or clusters centroid).


### Metrics of clusterisation quality

* Intra-cluster distance ("distortion cost")

$\sum_{k=1}^{K} \sum_{i=1}^{l} [Q(x_i) = C_k] \cdot \rho(x_i; c_k) \rightarrow min$,

where $[Q(x_i) = k]$ checks whether $x_i$ belongs to k-th cluster $C_k$ and $\rho$ is the distance from the i-th object, $x_i$, to the center, $c_k$, of the k-th cluster. For example, we could minimize the pairwise squared deviations of points in the same cluster:

$\sum_{k=1}^{K} \sum_{i\in C_k} |x_i - c_k|^2 \rightarrow min$


### Exercise

For this part of the project you will compare clustering obtained via K-means to the (soft) clustering induced by EM. In order to do so, our K-means algorithm will differ a bit from the one you learned. Here, the means are estimated exactly as before but the algorithm returns additional information. More specifically, we use the resulting clusters of points to estimate a Gaussian model for each cluster. Thus, our K-means algorithm actually returns a mixture model where the means of the component Gaussians are the 𝐾 centroids computed by the K-means algorithm. This is to make it such that we can now directly plot and compare solutions returned by the two algorithms as if they were both estimating mixtures.


Since the initialization is random, please use seeds 0,1,2,3,4 to and select the one that minimizes the total cost. 


```python
# import custom functions
import common as common
import kmeans as kmeans
```


```python
# import data
X = np.loadtxt("toy_data.txt")
```

## K = 1


```python
K=1
total_cost = []

for ii in range(0,5):
    gaussian_mix, post =  common.init(X=X,K=K,seed=ii)
    new_gaussian_mix, new_post, cost = kmeans.run(X=X,mixture=gaussian_mix,post=post)
    total_cost.append(cost)
    common.plot(X=X,mixture=new_gaussian_mix, post=new_post, title="K={}, seed ={}".format(str(K),str(ii)))

plt.plot(total_cost)
plt.xlabel("seed")
plt.ylabel("cost")
plt.show()

print("min cost for K = {} is ".format(str(K)), min(total_cost))
```


![png](project4-kmeans-em_files/project4-kmeans-em_8_0.png)



![png](project4-kmeans-em_files/project4-kmeans-em_8_1.png)



![png](project4-kmeans-em_files/project4-kmeans-em_8_2.png)



![png](project4-kmeans-em_files/project4-kmeans-em_8_3.png)



![png](project4-kmeans-em_files/project4-kmeans-em_8_4.png)



![png](project4-kmeans-em_files/project4-kmeans-em_8_5.png)


    min cost for K = 1 is  5462.297452340001


## K = 2


```python
K=2
total_cost = []

for ii in range(0,5):
    gaussian_mix, post =  common.init(X=X,K=K,seed=ii)
    new_gaussian_mix, new_post, cost = kmeans.run(X=X,mixture=gaussian_mix,post=post)
    total_cost.append(cost)
    common.plot(X=X,mixture=new_gaussian_mix, post=new_post, title="K={}, seed ={}".format(str(K),str(ii)))

plt.plot(total_cost)
plt.xlabel("seed")
plt.ylabel("cost")
plt.show()

print("min cost for K = {} is ".format(str(K)), min(total_cost))
```


![png](project4-kmeans-em_files/project4-kmeans-em_10_0.png)



![png](project4-kmeans-em_files/project4-kmeans-em_10_1.png)



![png](project4-kmeans-em_files/project4-kmeans-em_10_2.png)



![png](project4-kmeans-em_files/project4-kmeans-em_10_3.png)



![png](project4-kmeans-em_files/project4-kmeans-em_10_4.png)



![png](project4-kmeans-em_files/project4-kmeans-em_10_5.png)


    min cost for K = 2 is  1684.9079502962372


## K = 3


```python
K=3
total_cost = []

for ii in range(0,5):
    gaussian_mix, post =  common.init(X=X,K=K,seed=ii)
    new_gaussian_mix, new_post, cost = kmeans.run(X=X,mixture=gaussian_mix,post=post)
    total_cost.append(cost)
    common.plot(X=X,mixture=new_gaussian_mix, post=new_post, title="K={}, seed ={}".format(str(K),str(ii)))

plt.plot(total_cost)
plt.xlabel("seed")
plt.ylabel("cost")
plt.show()

print("min cost for K = {} is ".format(str(K)), min(total_cost))
```


![png](project4-kmeans-em_files/project4-kmeans-em_12_0.png)



![png](project4-kmeans-em_files/project4-kmeans-em_12_1.png)



![png](project4-kmeans-em_files/project4-kmeans-em_12_2.png)



![png](project4-kmeans-em_files/project4-kmeans-em_12_3.png)



![png](project4-kmeans-em_files/project4-kmeans-em_12_4.png)



![png](project4-kmeans-em_files/project4-kmeans-em_12_5.png)


    min cost for K = 3 is  1329.5948671544297


## K = 4


```python
K=4
total_cost = []

for ii in range(0,5):
    gaussian_mix, post =  common.init(X=X,K=K,seed=ii)
    new_gaussian_mix, new_post, cost = kmeans.run(X=X,mixture=gaussian_mix,post=post)
    total_cost.append(cost)
    common.plot(X=X,mixture=new_gaussian_mix, post=new_post, title="K={}, seed ={}".format(str(K),str(ii)))

plt.plot(total_cost)
plt.xlabel("seed")
plt.ylabel("cost")
plt.show()

print("min cost for K = {} = ".format(str(K)), min(total_cost))
```


![png](project4-kmeans-em_files/project4-kmeans-em_14_0.png)



![png](project4-kmeans-em_files/project4-kmeans-em_14_1.png)



![png](project4-kmeans-em_files/project4-kmeans-em_14_2.png)



![png](project4-kmeans-em_files/project4-kmeans-em_14_3.png)



![png](project4-kmeans-em_files/project4-kmeans-em_14_4.png)



![png](project4-kmeans-em_files/project4-kmeans-em_14_5.png)


    min cost for K = 4 is  1035.4998265394659


# EM algorithm

### Exercise

Generate analogous plots to K-means using your EM implementation. Note that the EM algorithm can also get stuck in a locally optimal solution. For each value of 𝐾, please run the EM algorithm with seeds 0,1,2,3,4 and select the solution that achieves the highest log-likelihood. Compare the K-means and mixture solutions for 𝐾=[1,2,3,4]. Ask yourself when, how, and why they differ.


```python
import em as em
```


```python
mixture, post = common.init(X, 2, 0)
_ = em.run(X, mixture,post)
```

# K = 1 


```python
K=1
total_cost_em = []
total_cost_km = []

for ii in range(0,5):
    gaussian_mix, post =  common.init(X=X,K=K,seed=ii)
    new_gaussian_mix_em, new_post_em, cost_em = em.run(X=X,mixture=gaussian_mix,post=post)
    total_cost_em.append(cost_em)
    new_gaussian_mix_km, new_post_km, cost_km = kmeans.run(X=X,mixture=gaussian_mix,post=post)
    total_cost_km.append(cost_km)
    common.plot_both(X=X,
                    mixture_km=new_gaussian_mix_km, post_km=new_post_km, title_km="k-means algorithm, K={}, seed ={}".format(str(K),str(ii)),
                    mixture_em=new_gaussian_mix_em, post_em=new_post_em, title_em="EM algorithm, K={}, seed ={}".format(str(K),str(ii)))
plt.plot(total_cost_em)
plt.xlabel("seed")
plt.ylabel("cost EM")
plt.show()
plt.plot(total_cost_km)
plt.xlabel("seed")
plt.ylabel("cost k-means")
plt.show()

print("EM max likelihood for K = {} = ".format(str(K)), max(total_cost_em))
print("k-means min cost for K = {} = ".format(str(K)), min(total_cost_km))
```


![png](project4-kmeans-em_files/project4-kmeans-em_20_0.png)



![png](project4-kmeans-em_files/project4-kmeans-em_20_1.png)



![png](project4-kmeans-em_files/project4-kmeans-em_20_2.png)



![png](project4-kmeans-em_files/project4-kmeans-em_20_3.png)



![png](project4-kmeans-em_files/project4-kmeans-em_20_4.png)



![png](project4-kmeans-em_files/project4-kmeans-em_20_5.png)



![png](project4-kmeans-em_files/project4-kmeans-em_20_6.png)


    EM max likelihood for K = 1 =  -1307.2234317600942
    k-means min cost for K = 1 =  5462.297452340001


### k-means and EM-algorithm comparison for K=2: 

he results for 𝐾=1 case are exactly the same for k-ameans and EM-algorithm. All the data are assigned to the same cluster.

# K = 2 


```python
K=2
total_cost_em = []
total_cost_km = []

for ii in range(0,5):
    gaussian_mix, post =  common.init(X=X,K=K,seed=ii)
    new_gaussian_mix_em, new_post_em, cost_em = em.run(X=X,mixture=gaussian_mix,post=post)
    total_cost_em.append(cost_em)
    new_gaussian_mix_km, new_post_km, cost_km = kmeans.run(X=X,mixture=gaussian_mix,post=post)
    total_cost_km.append(cost_km)
    common.plot_both(X=X,
                    mixture_km=new_gaussian_mix_km, post_km=new_post_km, title_km="k-means algorithm, K={}, seed ={}".format(str(K),str(ii)),
                    mixture_em=new_gaussian_mix_em, post_em=new_post_em, title_em="EM algorithm, K={}, seed ={}".format(str(K),str(ii)))
plt.plot(total_cost_em)
plt.xlabel("seed")
plt.ylabel("cost EM")
plt.show()
plt.plot(total_cost_km)
plt.xlabel("seed")
plt.ylabel("cost k-means")
plt.show()

print("EM max likelihood for K = {} = ".format(str(K)), max(total_cost_em))
print("k-means min cost for K = {} = ".format(str(K)), min(total_cost_km))
```


![png](project4-kmeans-em_files/project4-kmeans-em_23_0.png)



![png](project4-kmeans-em_files/project4-kmeans-em_23_1.png)



![png](project4-kmeans-em_files/project4-kmeans-em_23_2.png)



![png](project4-kmeans-em_files/project4-kmeans-em_23_3.png)



![png](project4-kmeans-em_files/project4-kmeans-em_23_4.png)



![png](project4-kmeans-em_files/project4-kmeans-em_23_5.png)



![png](project4-kmeans-em_files/project4-kmeans-em_23_6.png)


    EM max likelihood for K = 2 =  -1175.7146293666792
    k-means min cost for K = 2 =  1684.9079502962372


### k-means and EM-algorithm comparison for K=2: 

Resulting clusters are very similar in terms of their mean and point assignment. Because the EM algorithm uses soft  assignment model, the points in the middle belong to both clusters, just with different probabilities.

# K = 3


```python
K=3
total_cost_em = []
total_cost_km = []

for ii in range(0,5):
    gaussian_mix, post =  common.init(X=X,K=K,seed=ii)
    new_gaussian_mix_em, new_post_em, cost_em = em.run(X=X,mixture=gaussian_mix,post=post)
    total_cost_em.append(cost_em)
    new_gaussian_mix_km, new_post_km, cost_km = kmeans.run(X=X,mixture=gaussian_mix,post=post)
    total_cost_km.append(cost_km)
    common.plot_both(X=X,
                    mixture_km=new_gaussian_mix_km, post_km=new_post_km, title_km="k-means algorithm, K={}, seed ={}".format(str(K),str(ii)),
                    mixture_em=new_gaussian_mix_em, post_em=new_post_em, title_em="EM algorithm, K={}, seed ={}".format(str(K),str(ii)))
plt.plot(total_cost_em)
plt.xlabel("seed")
plt.ylabel("cost EM")
plt.show()
plt.plot(total_cost_km)
plt.xlabel("seed")
plt.ylabel("cost k-means")
plt.show()

print("EM max likelihood for K = {} = ".format(str(K)), max(total_cost_em))
print("k-means min cost for K = {} = ".format(str(K)), min(total_cost_km))
```


![png](project4-kmeans-em_files/project4-kmeans-em_26_0.png)



![png](project4-kmeans-em_files/project4-kmeans-em_26_1.png)



![png](project4-kmeans-em_files/project4-kmeans-em_26_2.png)



![png](project4-kmeans-em_files/project4-kmeans-em_26_3.png)



![png](project4-kmeans-em_files/project4-kmeans-em_26_4.png)



![png](project4-kmeans-em_files/project4-kmeans-em_26_5.png)



![png](project4-kmeans-em_files/project4-kmeans-em_26_6.png)


    EM max likelihood for K = 3 =  -1138.8908996872672
    k-means min cost for K = 3 =  1329.5948671544297


### k-means and EM-algorithm comparison for K=3: 

The clusters delivered by the k-means algorithm are grouped so as to minimize the intra-cluster distance metric (distortion cost). 

On the contrary, the left two clusters delivered by the EM algorithm are closely packed with very different variances. This is probably because the EM algoroithm accounts more for the higher density of points on the left side. 

# K = 4


```python
K=4
total_cost_em = []
total_cost_km = []

for ii in range(0,5):
    gaussian_mix, post =  common.init(X=X,K=K,seed=ii)
    new_gaussian_mix_em, new_post_em, cost_em = em.run(X=X,mixture=gaussian_mix,post=post)
    total_cost_em.append(cost_em)
    new_gaussian_mix_km, new_post_km, cost_km = kmeans.run(X=X,mixture=gaussian_mix,post=post)
    total_cost_km.append(cost_km)
    common.plot_both(X=X,
                    mixture_km=new_gaussian_mix_km, post_km=new_post_km, title_km="k-means algorithm, K={}, seed ={}".format(str(K),str(ii)),
                    mixture_em=new_gaussian_mix_em, post_em=new_post_em, title_em="EM algorithm, K={}, seed ={}".format(str(K),str(ii)))
plt.plot(total_cost_em)
plt.xlabel("seed")
plt.ylabel("cost EM")
plt.show()
plt.plot(total_cost_km)
plt.xlabel("seed")
plt.ylabel("cost k-means")
plt.show()

print("EM max likelihood for K = {} = ".format(str(K)), max(total_cost_em))
print("k-means min cost for K = {} = ".format(str(K)), min(total_cost_km))
```


![png](project4-kmeans-em_files/project4-kmeans-em_29_0.png)



![png](project4-kmeans-em_files/project4-kmeans-em_29_1.png)



![png](project4-kmeans-em_files/project4-kmeans-em_29_2.png)



![png](project4-kmeans-em_files/project4-kmeans-em_29_3.png)



![png](project4-kmeans-em_files/project4-kmeans-em_29_4.png)



![png](project4-kmeans-em_files/project4-kmeans-em_29_5.png)



![png](project4-kmeans-em_files/project4-kmeans-em_29_6.png)


    EM max likelihood for K = 4 =  -1138.6011756994856
    k-means min cost for K = 4 =  1035.4998265394659


The same reasoning as for the K=3 case.
