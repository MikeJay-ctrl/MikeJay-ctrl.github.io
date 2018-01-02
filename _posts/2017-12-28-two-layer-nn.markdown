---
layout: post
title:  "Two layer NN - Numpy Implementation - MNIST Classifier"
date:   2017-12-28 19:00:24 +0000
category: "Neural Networks"
---



**When I first got started with machine learning tensorFlow didnt exist! (at least publicly)**

When I was completing my masters, all equations and transformations had to be intimately understood, in order to form any kind of working implementation of a machine learning algorithm. 

TensorFlow is a great tool and i'm glad it exists now, however I really do believe that to be the best at machine learning, you need to understand whats going on under the hood.

While I dont plan to go into the mathematics here, in this post I will create a simple 2 layer neural network (with backprop) and use it to classify the MNIST dataset. In order to give some intuition as to what is going on behind the scenes with tensorflow.


The code is available from my Github.
I'll do my best to keep it in a single file and explain as much as possible.  

<br/><br/><br/>

## Step 1: Import Data

{% highlight ruby %}

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline

{% endhighlight %}
These are all the libraries we will be using for today.
* The **mnist** library allows us to extract the MNIST dataset from the zipped files. 
* **Numpy** is the de-facto standard for Machine Learning "tensor" (vector) manipulation.
* **MatplotLib** will allow us to visualise our data.

The last line simply tells us to display our plots inline


{% highlight ruby %}
data = MNIST('../data')
train_images, train_labels = data.load_training()
test_images, test_labels = data.load_testing()

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
{% endhighlight %}

This allows us to import the MNIST dataset from using the mnist libry, This will automatically extract the data from the zipped files and load the images and labels into memory.

<br/><br/><br/>

## Step 2: Data and pre-processing

We can run some checks on this data to make sure it's as we expect.

{% highlight ruby %}
## Sanity check

print("analyse loaded data:\n ")
print(" Training images - {}\n Training labels - {}".format(len(train_images), len(train_labels))) 
print(" Test images - {}\n Test labels - {}".format(len(test_images), len(test_labels))) 

## train labels are stored as a single vector with the number corisponding to which digit the image is
## test labels are stored as a 784 index 1d array


# visualise image data
samples = np.random.randint(60000, size=(5))


for idx, sample in enumerate(samples):
    image = np.asarray(train_images[sample]).reshape(28, 28)
    label = train_labels[sample]
    plt.subplot(1, 5, idx+1)
    plt.imshow(image, cmap='gray')
    plt.title('label = {}'.format(label))
    plt.axis('off')
    
{% endhighlight %}

The output should be similar to what is shown below.. however the numbers will not be identical because they are randomly chosen.
<br/><br/>
![graph](/images/img/2017/12/sample-digits.png)


{% highlight ruby %}
# make labels one-hot

train_labels = np.array(train_labels)

train_labels_v = np.zeros((60000,10))
train_labels_v[np.arange(60000), train_labels] = 1

test_labels = np.array(test_labels)

test_labels_v = np.zeros((10000,10))
test_labels_v[np.arange(10000), test_labels] = 1
{% endhighlight %}

We want to make our vectors one-hot encoded.. one hot encoding means that we represent each of the available classes in our dataset with one long array. The length of this array is equal to the number of classes we have to choose from.

The values in this array are all set equal to zero, the class which correctly classifies a particular sample will be set to 1.

In the above, we make an array with the same number rows as we have sample images. Each row will have 10 columns, the same number of columns as we have classes. 10 columns represent numbers 0-9.

We then iterate through each of our newly made vectors and for each row in our sample data set (60000 for train and 10000 for test) we take the label index( which will be a number from 0-9) and set the value at that position in the array to to 1.

<br/><br/><br/>
## Step 3: Define network model

OK.. ready..


first things first, we create a class to encapsulate our model and its parameters, i've called it NNModel (Neural Network Model).

This model take 3 initialization parametars:

* **input_sz** - the number of input images in the current batch for (10000 for test batch)
* **hidden_sz** - the number of neurones in the hidden layer
* **output_sz** - the total number of classes that we can mapp to
<br/><br/>

{% highlight ruby %}
class NNModel:
	def __init__(self, input_sz, hidden_sz, output_sz,weightInitFcn=None, std=1e-4)
       	self.W1 = std * np.random.randn(input_sz, output_sz) 
       	self.b1 = np.zeros(output_sz) 
{% endhighlight %}
<br/>
In the initialization class abovewe initialize the weights W1 and the bias b1 as parameters, making them class variables using the 'self.' prefix.

* **W1** is 2d matrix with dimensions [input_sz * output_sz]
* **b1** is 1d column vector of length output_sz



The calc_loss method is going to hold the bulk of our ML logic,
for brevity, I've put the forward and backward pass within the same function. You would definitely factor these steps out in commercial code or a larger network.

Lets look at the forward pass first:

{% highlight ruby %}
def calc_loss(self, x, y, y_gt, reg):

    #forward pass
    scores = x.dot(self.W1) + self.b1
        
{% endhighlight %}

Very simply, we take x and calculate the dot product between it and the weights matrix initialised earlier, we then add the bias to ths to produce a1.
a1 is essentially our best guess, given each image, of what number this image depicts (which class 0-9 it belongs to)


<br/><br/><br/>
## Step 4: Loss


#### **4.1 - Define loss metrics**
  We need to calculate the loss associated with that forward pass, i.e how (in)accurate are our predictions.

In order to do this we will use two functions, the first is the **Softmax** function.
<br/><br/>
{% highlight ruby %}
def softmax(self, scores):
    scores -= np.max(scores, axis=1, keepdims=True)
    e_scores = np.exp(scores)
    return e_scores/ np.sum(e_scores, axis=1, keepdims=True) 
{% endhighlight %}
<br/>
The softmax function takes our uncorrelated scores and converts them into a probability distribution over all classes, its considered a probability because all values are positive and together sum to 1.

Next we calculate the cross entropy loss with respect to our data.
<br/><br/>
{% highlight ruby %}
def c_e_loss(self, probs, labels):
    N = probs.shape[0]
    return (np.sum(-np.log(probs[range(N), labels])) / N)
{% endhighlight %}
<br/>
The **Cross entropy** loss also known as the negative log likelihood, is used to measure the (dis)similarity between the true class and the predicted class.

Once we have defined our data loss we include a regularization term in order to try and coherse our function to favour simpler (lower order) predictions.
This is so called 'L2 Regularization'. It essentially means that if the same function is represented by a higher and a lower order equation, due to the magnitude of the higher order function having a greater L2 norm (which is in turn added to the overall loss), the lower order function will be prefered.


The loss calculation section of our forward pass is shown below
<br/><br/>
{% highlight ruby %}
#loss calculation	
probabilities = self.softmax(scores)

data_loss = self.c_e_loss(probabilities, y_gt)
reg_loss = 0.5 * reg *np.sum(self.W1*self.W1)       
loss = data_loss + reg_loss
{% endhighlight %}


<br/>
#### **4.2 - Backpropagation**

Finally, we need to backpropagate in order to update and adjust our weights. This means that the next time we make a prediction, it will be slightly better than what we have estimated previously.

In order to do this we need to understand how much effect each value used, at each stage of our forward pass, had on our final outcome. Once we know this we can adjust these values in a way that will minimise the loss incurred.

The code for the backward pass (backpropagation) is shown below.

{% highlight ruby %}
#backward pass  -- gradient
dscores = probabilities
dscores[range(N), y_gt] -= 1
dscores /= N

db1 = np.sum(dscores, axis=0)
dw1 = np.dot(x.T, dscores)
dx = np.dot(dscores, self.W1.T)
{% endhighlight %}

Because we only have one set of weights, really we only need to calculate the loss with respect to self.W1 and self.b1, however if this network were any deeper we would use the loss with respect to x as a parameter to calculate the loss with respect to earlier layers.


The process of calculating the derivative of the loss with respect to the set of parameters, and using that derivative iteratively to minimise future losses is called Gradient Descent.

<br/><br/><br/>
## Step 5: Train our model!

This is the Iterative part of the process eluded to in the previous paragraph


{% highlight ruby %}
def train(self, x, y, y_gt, lr, batch_size, iterations, reg=5e-6):
    num_train = x.shape[0]
    its_per_epoch = max(num_train/iterations, 1)
    loss_history = []
        
    for i in range(iterations):     
        samples = np.random.choice(np.arange(num_train), batch_size)

        x_batch = x[samples]
        y_batch = y[samples]
        y_gt_batch = y_gt[samples]
            
        loss, grads = self.calc_loss(x_batch, y_batch, y_gt_batch, reg)
        loss_history.append(loss)
        self.W1 += -lr * grads['W1']
        self.b1 += -lr * grads['b1']
            
        #if it % its_per_epoch == 0:
        #print something
            
    return {
        'loss_history' : loss_history
    }

{% endhighlight %}

Instead of trying to update all the weights of all the samples in what could be a dataset of millions of points we perform 'stocastic' gradient descent. Where we take a random subset of the samples and adjust the weights based on the results of our predictions on this subset.

Each subset is called a batch, you can see that we pass a batch of images and its corresponding labels to the calc_loss function and based on those results we update our weights.

We are returning our loss as well as the gradient so that we can plot our loss per iteration on a graph and make inferences based on its shape.

<br/><br/><br/>
## TEST *(Finally)*

Once we've finished training our model we use the predict function, on our test dataset to see how accurately we can predict each example image.

The code below performs training and test on the model defined above, the outcome is shown in the plot below.

{% highlight ruby %}
#use model on dataset

# model params
input_size = 784
hidden_size = 50
output_size = 10
learning_rate = 1e-6
batch_size = 300
iterations = 500

model = FFNNModel(input_size, hidden_size, output_size)
stats = model.train(train_images, train_labels_v, train_labels, lr=learning_rate, batch_size=batch_size, iterations=iterations)

#validation_res = model.predict(validation_images)
test_res = model.predict(test_images)
#review validation results
#--- here ---#
#accuracy = np.equal(test_res[np.arange(test_labels.shape[0]), test_labels_v] == 1)
accuracy = (test_res == test_labels).mean()
print(accuracy)
#test model

#review test results
#--- here ---#
#print(stats)
plt.plot(stats['loss_history'])
plt.xlabel('its')
plt.show()

{% endhighlight %}

![graph](/images/img/2017/12/error-per-iteration.png)

#### This simple model is able to achieve around 90% accuracy!


