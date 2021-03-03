
# My notes and thoughts

## Tips and tricks

### Prevent Colab from disconnecting by adding this code to the browser's console
* function ClickConnect(){
    console.log("Clicked on connect button"); 
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)

## Normalization
### Batch normalization
* Normalizing  the input. Is it important? Why? Connection to the initialization?


## Initialization
* If the weights are all the same, all neurons will learn the same things. 
* Hence initializing all weights to 0 or to a constant makes no sense.
* **Effective randomization of weights is crucial to learning good mappings from input to output**
* Too small weights >>> y = a[L] = W[L] small_number^{L-1} x >>> Vanishing gradient problem >>> Slow learning
* Too large weights >>> y = a[L] = W[L] large_number^{L-1} x >>> Exploding gradient problem >>> Divergence, the cost oscillates around its minimum.

Example: normal distribution of weights mean = 0, std = 1.0

 <img src="https://render.githubusercontent.com/render/math?math=input=X_1W_1+X_2W_2+...">
 
where Xs are the input neurons and Ws are the respective weights. The variance of each element in this sum is 

 <img src="https://render.githubusercontent.com/render/math?math=Var(X_iW_i)= E[X_i^2W_i^2] - E[X_iW_i]^2  = [E(X_i)]^2Var(W_i)  %2B  [E(W_i)]^2Var(X_i) %2B Var(X_i)Var(W_i)">
 
Under the assumption the input was scaled with the mean = 0 and a unit variance,  the variance is = 1. Hence, the total variance *Var(input) = n*, where *n* is the number of inputs. For large *n*, the variance of the input will be large too and this will lead to a saturation of the activation function already in the first layer :  *z_1 = tanh (XW) =  tanh(input) >> |2| *

* The variance of the weights must be kept constant throughout the training. This is possible to achieve with, for example, Xavier's and He's initializations.

### Xavier initialization  
[Glorot 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi)
* Works with **sigmoid and tanh activations**
* Zero mean and constant variance of the weights across all layers.
* Initialize the weights randomly from a **normal distribution*** with mean = 0 and variance <img src="https://render.githubusercontent.com/render/math?math=\sigma=\sqrt{6/(nin + nout)}">  , where *nin* and *nout* are the average number of the input and output neurons. 
* Assumes that the activations are linear.
* This prevents both the vanishing and exploding gradients problems

```tf.keras.initializers.GlorotUniform()```

### He initialization
[He 2015](https://arxiv.org/pdf/1502.01852.pdf)
* For **ReLU activation** a common initialization is He initialization
* In Xavier initialization, the activations are assumed to be linear (tanh(z) = z, as the weights are small). This does not work for the ReLU activations.
* 



## Data sets

### Train-test-validate
* Split data into training, validation (optional) and testing

* If no validation:  *Apply 1x to test set
* If validation: *Apply to test set and refine, apply 1x to validation

### Size
* Avoid small sample sizes
**Large samples : 60%-train /  20%-test / 20%-validate
**Medium sample size: 60%-train / 40%-test
**Small sample size: Do cross-validation, report caveat of small sample size

### Tips
* Set test/validation set saside
* Randomly sample training and test
* As much diversity as possible - random assignment does this 

## Prediction errors and probabilities

* In-sample error 
* Out-of-sample error = generelazitation error

**Example: Is it a dog or not?**

* P probability
* + positive test
* - negative test
* D subject is a dog
* Dc subject is NOT a dog

### Test results, specificity, sensitivity and accuracy of the test
* True positive - it is a dog, and the model confirms this is a dog
* False positive - it is NOT a dog, but the model confirms this is a dog
* True negative - it is NOT a dog and the model rejects this is a dog
* False negative - it is a dog, but the model rejects this is a dog

* Sensitivity = P(+|D) = TP / (TP + FN)   - if you want few missed positives
* Specificity = P(-|Dc) = TN / (FP + TN)   -  MUST be HIGH for test to be good - if you want few negatives called positives
* Posititve Predictive Value = P(D|+) = TP / (TP + FP)
* Negative Predictive Value = P(Dc|-) = TN / (FN + TN)
* Accuracy = P(correct outcome) = (TP+TN) / (TP + FP + FN + TN)   - weights false positives and negatives equally 

### **Bayes formula** (probability of being a dog given a positive test result)

 <img src="https://render.githubusercontent.com/render/math?math=P(D|%2B)=\frac{P(%2B|D)P(D)}{P(%2B|D)P(D) %2B P(%2B|Dc)P(Dc)}">

 * where we can replace
 * P+|Dc) = 1 - P(-|Dc)
 * P(Dc) = 1 - P(D)

* Lets consider a pregnancy test. The sensitivity of the test, P(+|pregnant), is 75%, the specificity, P(-|not pregnant), is 52% and the prevalence of being pregnant, P(pregant), is 30% (30% of women taking pregnancy tests are actually pregnant). What is a probability of being pregnant (P) having a positive test result (+)? Following the Bayes formula, P(pregnant|+) =  P(+|pregnant)P(pregnant) / (P(+|pregnant)P(pregnant) + *P(+|not pregnant) P (not pregnant))* = P(+|pregnant)P(pregnant) / (P(+|pregnant)P(pregnant) + *1-P(-|not pregnant) (1-P (pregnant)))* = 0.75 * 0.3 / (0.75 * 0.3 + 0.48  * 0.7) = 0.4


* Mean squared error (or root mean sq. error) - continuous data, sensitive to outliers
* Median absolute deviation - cont. data, often more robust

## Cross-validation
* k-fold CV:
- large k = low bias, high variance ("overfitting")
- small k = high bias, low variance ("underfitting")

* **For cross validation,  the data MUST be picked by random sampling without replacement** (if you picked no. 1 you will not going to pick it again until you gone thru the whole dataset - *COMPARE to SGD where the option with replacement is preferrable!*)
* For CV, picking data **with replacement** will result in an **underestimation of an error** (apparently because we may randomly pick the same data again and again)

## Underfitting and overfitting
### Underfitting 
* Underfitting means the model was not trained sufficiently long and performs poorly both on train and validation data.
* An underfit model is characterised by **high bias and low variance**. 
* Underfitting can be adressed by increasing the capacity of the model  (an ability to fit a variety of different functions)
* For example,  by increasing the number of hidden layers and (or) the number of nodes in them.

### Overfitting
* Data have signal and noise
* One can always design a perfect in-sample predictor, you capture bioth signal and noise when you do that. This predictor will not perform well on out of sample data!
* Overfitting means a model performs very well on train data and poorly on validation data. 
* An overfit model is characterised by **low bias and high variance**. Its performance **varies** strongly for unseen data. And it performs very well on the seen data.
* The model is "too complicated" for this dataset and hence learnt features which do not exist in reality.



There are two way to address overfitting:
1. Increase the amount of data used for training.  
  >>> **Grid search**
  >>> Removal of excess nodes
2. Reduce the complexity of the model: 
  >>> By decreasing the number of adaptive parameters (layers and nodes) in the network (this is called structural stabilization). 
  >>> Or by controlling the complexity of a model through the use of regularization (via addition of a penalty term to the error function, which encourages the weights to be small).
  
## Data augmentation
* [Hernández-García 2018](https://arxiv.org/abs/1806.03852) systematically analyzed the effect of data augmentation on some popular architectures and conclude that data augmentation alone—without any other explicit regularization techniques—can achieve the same performance or higher as regularized models, especially when training with fewer examples.


## Optimization algorithms 

* For a given NN architecture, the values of parameters determine the performance of a model.
* The goal is to find parameters that match predictions with the ground truth as close as possible.
* This is done by defining a loss function and minimizing it.
* The way the loss function is defined determined the performance of the model.

* The cost function is  the average of the loss *L* computed for the entire training data set:

<img src="https://render.githubusercontent.com/render/math?math= J = 1/m \sum_{i=1}^m L^{i}"> 

* The cost function has a landscape that varies as a function fo the parameters of the NN.
* The goal is to find a point where the cost is (approximately) minimal.

* Depening on the amount of data, there is a trade-off between the accuracy and the computational complexity.

### Gradient descent

* Update parameters *w* as 

<img src="https://render.githubusercontent.com/render/math?math=w = w - \alpha \nabla J "> ,

where *J* is the cost function we are seeking to minimize,  *\nabla J = dJ/dw* is the gradient of the cost function computed w.r.t. the parameters *w* and  *\alpha* is the learning rate. In general, *J*  can be expressed as a **finite sum** of loss functions *L* computed for each training example *i*

<img src="https://render.githubusercontent.com/render/math?math= J(w) = \frac{1}{m} \sum_{i=1}^{m} L_i(w)">, 

where *m* is the number of training examples. 

* When *J* is computed for the whole  training data set (for all training examples), then the algorithm is sometimes called "batch gradient descent". 
* When the *J* is computed for a subset of training data and this *J* is used to update parameters of the whole model (consisting of *m* datasets), the algorithm is called "mini-batch GS".

* The learning rate *\alpha* determines the speed of convergence. If it is too small, the algorithm will converge slowly.
* If the LR is too large, the cost function will oscillate, the algorithm won't converge.
* Adaptive learning-rate algorithms such as Adam and RMSprop adjust the LR in the course of the optimization.

### Batch gradient descent
* May stuck in local minima

* Uses the whole dataset to compute the gradient used to update the model's parameters.
* The parameters are updated as follows:

<img src="https://render.githubusercontent.com/render/math?math=w=w - \alpha  \frac{1}{m} \sum_{i=1}^{m}   \nabla L_i(w)"> 

* Thus it is rather slow
* **Guaranteed to converge to a global min for convex error surafces, and to a local min for non-convex ones.**
* No online update of the model (with the new examples on-the-fly)


### Stochastic gradient descent
* MIT 18.065 Lecture on SGD https://www.youtube.com/watch?v=k3AiUhwHQ28
* Escapes local minima 
* The gradient of the function L(w) = sum(L_i(w)) is expensive to compute.
* Hence we approximate this gradient by a **randomized version**. 
* Depending on how good this randomized version is, the alg. may or may not converge to a right minimum.

* Update weights by computing the gradient of a stochastically chosen function L_i(w)
* Makes a very fast progress in the very beginning.
* In the very beginning, parameter values *w* are very far away from the optimum. 
* It turns out that **the full gradient and the stochastic gradient  have the same sign**. 
* **Hence computing only the stochastic gradient allows to make an update in the same direction.**
* "The region of confusion" *[w_min, w_max]* is a range of the parameter values *w* that minimize  the loss functions L_i(w).
* Because in SGD we compute only a randomly chosen *L_i(w)*, it is not possible to tell where the minimum of the function *L(w) = sum(L_i(w))* is! It can be anywhere between  *[w_min, w_max]*. Hence the value of the full gradient's minimum fluctuates  in this range (this is why the SGD oscillates like craze in the vicinity of the minimum)!
 
What do we mean by a randomized version? What is the randomness here?  

1. *The random case (the so-called with resplacement option)*: At each new iteration pick a new dataset from your training data according to a uniform probability distribution (means each time you access the memory with your entire dataset!). 
2. *The cycle  case (the so-called without replacement option)* Datasets are picked sequentially from the randomly shuffled training set (if you picked no. 1 you will not going to pick it again until you gone thru the whole dataset). 
2. *The shuffle  case (the so-called without replacement option)* Datasets are picked sequentially from the randomly shuffled training set  (if you picked no. 1 you will not going to pick it again until you gone thru the whole dataset) and the training set is shuffled before each pass 
**Modern ML libraries like tensorflow use option 2 or 3**.

### Mini-batch gradient descent 
* Performs the parameter update using the gradient computed  for  a set of training examples 
* Averages out the noise of the cost function of the SGD
* May even outperform the full GD as the original full batch may not reflect the data that well (because the available "full" dataset is only an aproximation to the actual data)

From the lecture by Prof. Gilbert Strang, MIT, 2018.

GD with monentum term:
<img src="https://render.githubusercontent.com/render/math?math=x_{k+1}=x_k -s\cdot z_k"> 

<img src="https://render.githubusercontent.com/render/math?math=z_k =  \nabla f_k + \beta z_{k-1} "> 

the equaltion becomes 2nd order diff. eq.


* Supposed to take longer if we extend out training dataset by replicating the data [Bishop, p. 264](https://www.amazon.com/Networks-Recognition-Advanced-Econometrics-Paperback/dp/0198538642)

## Batch size
* [Masters 2018](https://arxiv.org/abs/1804.07612) showed that the best performance is for mini-batch sizes between 2 and 32.
Other research on this topic:
* [Smith 2018](https://arxiv.org/pdf/1711.00489.pdf) - increase the learning rate and scale the batch size (tested for SGD, SGD with momentum, Nesterov momentum and Adam).
* [Hoffer 2017](https://papers.nips.cc/paper/2017/file/a5e0ff62be0b08456fc7f1e88812af3d-Paper.pdf) - 
* [Goyal 2018](https://arxiv.org/pdf/1706.02677.pdf) - No loss of accuracy whilst training large batches up to 8192 by a hyperparameter-free linear scaling of the learning rate, "warmup" scheme.



# Useful resources

### Mathematics

[**Linear algebra with ODEs**](http://faculty.bard.edu/belk/math213/) 

[**Proofs in mathematics**](https://www.cut-the-knot.org/proofs/index.shtml) 


### Data science
#### Lectures
[**Fundamentals of Data Science Stanford MS&E 226**](https://web.stanford.edu/class/msande226/l_notes.html) 

#### Other resources
[**Training big data**](https://gdmarmerola.github.io/big-data-ml-training/) 
[**Processing big data in chunks**](https://www.dataquest.io/m/164-processing-dataframes-in-chunks/) 



## Papers I read, sorted by date

### **12/2020 (2)**
* [Hernández-García 2018](https://arxiv.org/abs/1806.03852)
* [Bottou 2009](https://leon.bottou.org/publications/pdf/slds-2009.pdf) Curiously Fast Convergence of some Stochastic Gradient Descent Algorithms

## Papers I read, sorted by topic

### Data augmentation
* [Hernández-García 2018](https://arxiv.org/abs/1806.03852)
### Regularization
* [Hernández-García 2018](https://arxiv.org/abs/1806.03852)
### Gradient Descent
* [Bottou 2009](https://leon.bottou.org/publications/pdf/slds-2009.pdf) Curiously Fast Convergence of some Stochastic Gradient Descent Algorithms
