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

## Useful resources

### Mathematics

[**Linear algebra with ODEs**](http://faculty.bard.edu/belk/math213/) 

[**Proofs in mathematics**](https://www.cut-the-knot.org/proofs/index.shtml) 


### Data science
[**Fundamentals of Data Science Stanford MS&E 226**](https://web.stanford.edu/class/msande226/l_notes.html) 
