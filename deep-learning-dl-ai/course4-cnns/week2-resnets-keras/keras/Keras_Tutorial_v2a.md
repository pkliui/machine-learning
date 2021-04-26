# Keras tutorial - Emotion Detection in Images of Faces

Welcome to the first assignment of week 2. In this assignment, you will:
1. Learn to use Keras, a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK. 
2. See how you can in a couple of hours build a deep learning algorithm.

#### Why are we using Keras? 

* Keras was developed to enable deep learning engineers to build and experiment with different models very quickly. 
* Just as TensorFlow is a higher-level framework than Python, Keras is an even higher-level framework and provides additional abstractions. 
* Being able to go from idea to result with the least possible delay is key to finding good models. 
* However, Keras is more restrictive than the lower-level frameworks, so there are some very complex models that you would still implement in TensorFlow rather than in Keras. 
* That being said, Keras will work fine for many common models. 

## <font color='darkblue'>Updates</font>

#### If you were working on the notebook before this update...
* The current notebook is version "v2a".
* You can find your original work saved in the notebook with the previous version name ("v2").
* To view the file directory, go to the menu "File->Open", and this will open a new tab that shows the file directory.

#### List of updates
* Changed back-story of model to "emotion detection" from "happy house."
* Cleaned/organized wording of instructions and commentary.
* Added instructions on how to set `input_shape`
* Added explanation of "objects as functions" syntax.
* Clarified explanation of variable naming convention.
* Added hints for steps 1,2,3,4

## Load packages
* In this exercise, you'll work on the "Emotion detection" model, which we'll explain below. 
* Let's load the required packages.


```python
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline
```

**Note**: As you can see, we've imported a lot of functions from Keras. You can use them by calling them directly in your code. Ex: `X = Input(...)` or `X = ZeroPadding2D(...)`. 

In other words, unlike TensorFlow, you don't have to create the graph and then make a separate `sess.run()` call to evaluate those variables.

## 1 - Emotion Tracking

* A nearby community health clinic is helping the local residents monitor their mental health.  
* As part of their study, they are asking volunteers to record their emotions throughout the day.
* To help the participants more easily track their emotions, you are asked to create an app that will classify their emotions based on some pictures that the volunteers will take of their facial expressions.
* As a proof-of-concept, you first train your model to detect if someone's emotion is classified as "happy" or "not happy."

To build and train this model, you have gathered pictures of some volunteers in a nearby neighborhood. The dataset is labeled.
<img src="images/face_images.png" style="width:550px;height:250px;">

Run the following code to normalize the dataset and learn about its shapes.


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


**Details of the "Face" dataset**:
- Images are of shape (64,64,3)
- Training: 600 pictures
- Test: 150 pictures

## 2 - Building a model in Keras

Keras is very good for rapid prototyping. In just a short time you will be able to build a model that achieves outstanding results.

Here is an example of a model in Keras:

```python
def model(input_shape):
    """
    input_shape: The height, width and channels as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    """
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model
```

#### Variable naming convention

* Note that Keras uses a different convention with variable names than we've previously used with numpy and TensorFlow. 
* Instead of creating unique variable names for each step and each layer, such as 
```
X = ...
Z1 = ...
A1 = ...
```
* Keras re-uses and overwrites the same variable at each step:
```
X = ...
X = ...
X = ...
```
* The exception is `X_input`, which we kept separate since it's needed later.

#### Objects as functions
* Notice how there are two pairs of parentheses in each statement.  For example:
```
X = ZeroPadding2D((3, 3))(X_input)
```
* The first is a constructor call which creates an object (ZeroPadding2D).
* In Python, objects can be called as functions. Search for 'python object as function and you can read this blog post [Python Pandemonium](https://medium.com/python-pandemonium/function-as-objects-in-python-d5215e6d1b0d).  See the section titled "Objects as functions."
* The single line is equivalent to this:
```
ZP = ZeroPadding2D((3, 3)) # ZP is an object that can be called as a function
X = ZP(X_input) 
```

**Exercise**: Implement a `HappyModel()`.  
* This assignment is more open-ended than most. 
* Start by implementing a model using the architecture we suggest, and run through the rest of this assignment using that as your initial model. * Later, come back and try out other model architectures. 
* For example, you might take inspiration from the model above, but then vary the network architecture and hyperparameters however you wish. 
* You can also use other functions such as `AveragePooling2D()`, `GlobalMaxPooling2D()`, `Dropout()`. 

**Note**: Be careful with your data's shapes. Use what you've learned in the videos to make sure your convolutional, pooling and fully-connected layers are adapted to the volumes you're applying it to.


```python
# GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)
    #our input shape is n_W, n_H, n_C = (64,64,3)
    
    # Zero-Padding: pads the border of X_input with zeroes
    #X = ZeroPadding2D((2, 2))(X_input)
    # padding width 2p = 2*2 = 4
    
    # CONV -> BN -> RELU Block applied to X
    #X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    #the number of filters here is n_F = 32
    #here we set linear size of the conv. kernel f = 7
    #n_W' linear size of the conv layer is int (n_W + 2p - f) / s  + 1   =  int (64 + 2*2 - 7) / 1 +1 = 32
    #conv layer is 32,32,32 
    #
    #to obtain same number of pixels in the convolved data we use padding="same"
    X = Conv2D(32, (3, 3), strides = (1, 1), padding="same", name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    #stride is not specified, so it's by default = pool size
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    ### END CODE HERE ###
    
    return model
```

You have now built a function to describe your model. To train and test this model, there are four steps in Keras:
1. Create the model by calling the function above  

2. Compile the model by calling `model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])`  

3. Train the model on train data by calling `model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)`  

4. Test the model on test data by calling `model.evaluate(x = ..., y = ...)`  

If you want to know more about `model.compile()`, `model.fit()`, `model.evaluate()` and their arguments, refer to the official [Keras documentation](https://keras.io/models/model/).

#### Step 1: create the model.  
**Hint**:  
The `input_shape` parameter is a tuple (height, width, channels).  It excludes the batch number.  
Try `X_train.shape[1:]` as the `input_shape`.


```python
### START CODE HERE ### (1 line)
happyModel = HappyModel(X_train.shape[1:])
### END CODE HERE ###
```

#### Step 2: compile the model

**Hint**:  
Optimizers you can try include `'adam'`, `'sgd'` or others.  See the documentation for [optimizers](https://keras.io/optimizers/)  
The "happiness detection" is a binary classification problem.  The loss function that you can use is `'binary_cross_entropy'`.  Note that `'categorical_cross_entropy'` won't work with your data set as its formatted, because the data is an array of 0 or 1 rather than two arrays (one for each category).  Documentation for [losses](https://keras.io/losses/)


```python
### START CODE HERE ### (1 line)
happyModel.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ["accuracy"])
### END CODE HERE ###
```

#### Step 3: train the model

**Hint**:  
Use the `'X_train'`, `'Y_train'` variables.  Use integers for the epochs and batch_size

**Note**: If you run `fit()` again, the `model` will continue to train with the parameters it has already learned instead of reinitializing them.


```python
### START CODE HERE ### (1 line)
happyModel.fit(x = X_train, y = Y_train, epochs = 40, batch_size = 16)
### END CODE HERE ###
```

    Epoch 1/40
    600/600 [==============================] - 9s - loss: 0.7847 - acc: 0.7467     
    Epoch 2/40
    600/600 [==============================] - 8s - loss: 0.2118 - acc: 0.9217     
    Epoch 3/40
    600/600 [==============================] - 9s - loss: 0.1695 - acc: 0.9383     - ETA: 4s - loss: 0.
    Epoch 4/40
    600/600 [==============================] - 8s - loss: 0.0808 - acc: 0.9717     
    Epoch 5/40
    600/600 [==============================] - 8s - loss: 0.0867 - acc: 0.9633     
    Epoch 6/40
    600/600 [==============================] - 9s - loss: 0.1531 - acc: 0.9383     
    Epoch 7/40
    600/600 [==============================] - 8s - loss: 0.0473 - acc: 0.9917     
    Epoch 8/40
    600/600 [==============================] - 9s - loss: 0.0460 - acc: 0.9800     - ETA: 
    Epoch 9/40
    600/600 [==============================] - 9s - loss: 0.0347 - acc: 0.9883     
    Epoch 10/40
    600/600 [==============================] - 9s - loss: 0.0515 - acc: 0.9817     - ETA: 4s - loss: 0.0 - ETA: 0s - loss: 0.0551 - acc: 0.
    Epoch 11/40
    600/600 [==============================] - 9s - loss: 0.0326 - acc: 0.9900     
    Epoch 12/40
    600/600 [==============================] - 9s - loss: 0.1483 - acc: 0.9417     
    Epoch 13/40
    600/600 [==============================] - 9s - loss: 0.0358 - acc: 0.9900     
    Epoch 14/40
    600/600 [==============================] - 9s - loss: 0.2428 - acc: 0.9217     
    Epoch 15/40
    600/600 [==============================] - 9s - loss: 0.0561 - acc: 0.9833     
    Epoch 16/40
    600/600 [==============================] - 9s - loss: 0.0280 - acc: 0.9867     
    Epoch 17/40
    600/600 [==============================] - 9s - loss: 0.0430 - acc: 0.9850     
    Epoch 18/40
    600/600 [==============================] - 9s - loss: 0.0963 - acc: 0.9717     
    Epoch 19/40
    600/600 [==============================] - 9s - loss: 0.0485 - acc: 0.9850     
    Epoch 20/40
    600/600 [==============================] - 9s - loss: 0.0163 - acc: 0.9933     
    Epoch 21/40
    600/600 [==============================] - 9s - loss: 0.0221 - acc: 0.9967     - ETA: 6s
    Epoch 22/40
    600/600 [==============================] - 9s - loss: 0.0180 - acc: 0.9967     
    Epoch 23/40
    600/600 [==============================] - 9s - loss: 0.0158 - acc: 0.9950     
    Epoch 24/40
    600/600 [==============================] - 9s - loss: 0.0130 - acc: 0.9950     
    Epoch 25/40
    600/600 [==============================] - 9s - loss: 0.0129 - acc: 0.9967     
    Epoch 26/40
    600/600 [==============================] - 9s - loss: 0.0352 - acc: 0.9850     
    Epoch 27/40
    600/600 [==============================] - 9s - loss: 0.0174 - acc: 0.9917     
    Epoch 28/40
    600/600 [==============================] - 9s - loss: 0.0106 - acc: 0.9950     
    Epoch 29/40
    600/600 [==============================] - 9s - loss: 0.0218 - acc: 0.9900     - ETA: 5s - los
    Epoch 30/40
    600/600 [==============================] - 9s - loss: 0.0317 - acc: 0.9867     
    Epoch 31/40
    600/600 [==============================] - 9s - loss: 0.0284 - acc: 0.9933     
    Epoch 32/40
    600/600 [==============================] - 9s - loss: 0.0575 - acc: 0.9783     
    Epoch 33/40
    600/600 [==============================] - 9s - loss: 0.0190 - acc: 0.9917     
    Epoch 34/40
    600/600 [==============================] - 9s - loss: 0.0132 - acc: 0.9967     
    Epoch 35/40
    600/600 [==============================] - 9s - loss: 0.0043 - acc: 0.9983     
    Epoch 36/40
    600/600 [==============================] - 9s - loss: 0.0287 - acc: 0.9883     
    Epoch 37/40
    600/600 [==============================] - 9s - loss: 0.0080 - acc: 0.9950     
    Epoch 38/40
    600/600 [==============================] - 9s - loss: 0.0193 - acc: 0.9950     
    Epoch 39/40
    600/600 [==============================] - 9s - loss: 0.0497 - acc: 0.9900     - ETA: 0s - loss: 0.0508 - acc: 0.
    Epoch 40/40
    600/600 [==============================] - 9s - loss: 0.0144 - acc: 0.9933     





    <keras.callbacks.History at 0x7f43014388d0>



#### Step 4: evaluate model  
**Hint**:  
Use the `'X_test'` and `'Y_test'` variables to evaluate the model's performance.


```python
### START CODE HERE ### (1 line)
preds = happyModel.evaluate(x = X_test, y = Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
```

    150/150 [==============================] - 1s     
    
    Loss = 0.132035285036
    Test Accuracy = 0.953333330949


#### Expected performance   
If your `happyModel()` function worked, its accuracy should be better than random guessing (50% accuracy).

To give you a point of comparison, our model gets around **95% test accuracy in 40 epochs** (and 99% train accuracy) with a mini batch size of 16 and "adam" optimizer.

#### Tips for improving your model

If you have not yet achieved a very good accuracy (>= 80%), here are some things tips:

- Use blocks of CONV->BATCHNORM->RELU such as:
```python
X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)
```
until your height and width dimensions are quite low and your number of channels quite large (≈32 for example).  
You can then flatten the volume and use a fully-connected layer.
- Use MAXPOOL after such blocks.  It will help you lower the dimension in height and width.
- Change your optimizer. We find 'adam' works well. 
- If you get memory issues, lower your batch_size (e.g. 12 )
- Run more epochs until you see the train accuracy no longer improves. 

**Note**: If you perform hyperparameter tuning on your model, the test set actually becomes a dev set, and your model might end up overfitting to the test (dev) set. Normally, you'll want separate dev and test sets.  The dev set is used for parameter tuning, and the test set is used once to estimate the model's performance in production.

## 3 - Conclusion

Congratulations, you have created a proof of concept for "happiness detection"!

## Key Points to remember
- Keras is a tool we recommend for rapid prototyping. It allows you to quickly try out different model architectures.
- Remember The four steps in Keras: 


1. Create  
2. Compile  
3. Fit/Train  
4. Evaluate/Test  

## 4 - Test with your own image (Optional)

Congratulations on finishing this assignment. You can now take a picture of your face and see if it can classify whether your expression is "happy" or "not happy". To do that:


1. Click on "File" in the upper bar of this notebook, then click "Open" to go on your Coursera Hub.
2. Add your image to this Jupyter Notebook's directory, in the "images" folder
3. Write your image's name in the following code
4. Run the code and check if the algorithm is right (0 is not happy, 1 is happy)!
    
The training/test sets were quite similar; for example, all the pictures were taken against the same background (since a front door camera is always mounted in the same position). This makes the problem easier, but a model trained on this data may or may not work on your own data. But feel free to give it a try! 


```python
### START CODE HERE ###
img_path = 'images/my_image.jpg'
### END CODE HERE ###
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))
```

## 5 - Other useful functions in Keras (Optional)

Two other basic features of Keras that you'll find useful are:
- `model.summary()`: prints the details of your layers in a table with the sizes of its inputs/outputs
- `plot_model()`: plots your graph in a nice layout. You can even save it as ".png" using SVG() if you'd like to share it on social media ;). It is saved in "File" then "Open..." in the upper bar of the notebook.

Run the following code.


```python
happyModel.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_4 (InputLayer)         (None, 64, 64, 3)         0         
    _________________________________________________________________
    zero_padding2d_4 (ZeroPaddin (None, 68, 68, 3)         0         
    _________________________________________________________________
    conv0 (Conv2D)               (None, 62, 62, 32)        4736      
    _________________________________________________________________
    bn0 (BatchNormalization)     (None, 62, 62, 32)        128       
    _________________________________________________________________
    activation_3 (Activation)    (None, 62, 62, 32)        0         
    _________________________________________________________________
    max_pool (MaxPooling2D)      (None, 31, 31, 32)        0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 30752)             0         
    _________________________________________________________________
    fc (Dense)                   (None, 1)                 30753     
    =================================================================
    Total params: 35,617
    Trainable params: 35,553
    Non-trainable params: 64
    _________________________________________________________________



```python
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
```




![svg](Keras_Tutorial_v2a_files/Keras_Tutorial_v2a_30_0.svg)




```python

```
