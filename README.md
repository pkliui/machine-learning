# Independent coursework on machine learning

This documents records examples of my independent coursework on machine learning. 

## CNN in Tensorflow

### Image classification using a simple CNN model
[Cats_vs_Dogs_Classification_Simple_Model_Overfitting_Example]()
* Classification of cats and dogs images using a simple CNN model. An example which leads to overfiting.

[Cats_vs_Dogs_With_and_Without_Augmentation]()
* Classification of cats and dogs images with and without data augmentation.

###  Image classification using a CNN with data augmentation
[Course 2 - Exercise 2 - Cats vs. Dogs using augmentation]()
* Training a CNN on the full cats-vs-dogs dataset using data augmentation.



## My notes and thoughts



### Initialization
* If the weights are all the same, all neurons will learn the same things. 
* Hence initializing all weights to 0 or to a constant makes no sense.
* Too small weights >>> y = a[L] = W[L] small_number^{L-1} x >>> Vanishing gradient problem >>> Slow learning
* Too large weights >>> y = a[L] = W[L] large_number^{L-1} x >>> Exploding gradient problem >>> Divergence, the cost oscillates around its minimum.

#### Xavier initialization  
[Glorot 2010](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi)
* Works with **tanx activations**
* Zero mean and constant variance of the weights across all layers.
* Initialize the weights randomly from a *normal distribution* with mean = 0 and variance <img src="https://render.githubusercontent.com/render/math?math=\sigma=1/n^{l-1}">  , where n is the number of neurons in layer l-1. 
* Assumes that the activations are linear.
* This prevents both the vanishing and exploding gradients problems

For deep NNs:  ```tf.keras.initializers.GlorotUniform()```

#### He initialization
[He 2015](https://arxiv.org/pdf/1502.01852.pdf)
* For **ReLU activation** a common initialization is He initialization
* In Xavier initialization, the activations are assumed to be linear (tanh(z) = z, as the weights are small). This does nto work for the ReLU activations.


### Underfitting and overfitting
#### Underfitting 
* Underfitting means the model was not trained sufficiently long and performs poorly both on train and validation data.
* An underfit model is characterised by **high bias and low variance**. 
* Underfitting can be adressed by increasing the capacity of the model  (an ability to fit a variety of different functions)
* For example,  by increasing the number of hidden layers and (or) the number of nodes in them.

#### Overfitting
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
  
### Data augmentation
* [Hernández-García 2018](https://arxiv.org/abs/1806.03852) systematically analyzed the effect of data augmentation on some popular architectures and conclude that data augmentation alone—without any other explicit regularization techniques—can achieve the same performance or higher as regularized models, especially when training with fewer examples.


### Optimization algorithms 
* Depening on the amount of data, there is a trade-off between the accuracy and the computational complexity.

#### Batch gradient descent 
* Uses the whole dataset to perform one update
* Computes the gradient of the cost function w.r.t. the parameters theta:
<img src="https://render.githubusercontent.com/render/math?math=\theta=\theta - \eta \cdot \nabla_\theta J (\theta)"> 
* Thus very slow
* Guaranteed to converge to a global min for convex error  surafces, and to a local min for non-convex ones.
* No online update of the model (with the new examples on-the-fly)

#### Batch gradient descent 
* Performs a parameter update for each training example x(i) and label y(i):
<img src="https://render.githubusercontent.com/render/math?math=\theta=\theta - \eta \cdot \nabla_\theta J (\theta; x(i); y(i))"> 
* Supposed to take longer if we extend out training dataset by replicating the data [Bishop, p. 264](https://www.amazon.com/Networks-Recognition-Advanced-Econometrics-Paperback/dp/0198538642)


## Useful resources

### Mathematics

[**Linear algebra with ODEs**](http://faculty.bard.edu/belk/math213/) 

[**Proofs in mathematics**](https://www.cut-the-knot.org/proofs/index.shtml) 


### Data science
[**Fundamentals of Data Science Stanford MS&E 226**](https://web.stanford.edu/class/msande226/l_notes.html) 

## Papers I read, sorted by date

### **12/2020 (1)**
* [Hernández-García 2018](https://arxiv.org/abs/1806.03852)

## Papers I read, sorted by topic

### Data augmentation
* [Hernández-García 2018](https://arxiv.org/abs/1806.03852)
### Regularization
* [Hernández-García 2018](https://arxiv.org/abs/1806.03852)
