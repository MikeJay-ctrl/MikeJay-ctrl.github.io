---
layout: post
title:  "Two Layer Convolutional NN - Numpy Implementation"
date:   2017-12-30 08:00:24 +0000
category: "Neural Networks"
---


I recently did an implementation of a single layer Neural Network with backprop to classify the MNIST dataset

You can find that post [here]({{ site.url }}{% link _posts/2017-12-28-two-layer-nn.markdown %})

These days single layer networks are of limited use, deeper and more complex networks like the CNN are used to achieve lower errors and higher accuracy in image processing tasks.

In this post i will implement a CNN using just numpy. 


The code does run and is verified as working correctly.. **however the backpropagation is completely naive, unoptimised.. and for illustrative purposes only. It will take far too long to run on any real data.**

Once again, The code is available from my Github and I'll do my best to keep it in a single file and explain as much as possible.  

If you've read my previous post steps 1 and 2 will look familiar.


<br/><br/><br/>

## Step 1: Import Data

{% highlight ruby %}

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = MNIST('../data')
train_images, train_labels = data.load_training()
test_images, test_labels = data.load_testing()

{% endhighlight %}



{% highlight ruby %}
## sanity check 
#check the data was imported correctly as usual

print(' Number of training images - {}\n Number of training labels - {}'.format(len(train_images), len(train_labels)))
print(' Number of test images -{}\n Number of test labels - {}'.format(len(test_images), len(test_labels)))
{% endhighlight %}

* Number of training images - 60000
* Number of training labels - 60000
* Number of test images -10000
* Number of test labels - 10000


{% highlight ruby %}
## Visualise the data
samples = np.random.randint(60000, size=5)

for idx, sample in enumerate(samples):
    image = np.asarray(train_images[sample]).reshape(28, 28)
    label = train_labels[sample]
    
    plt.subplot(1, 5, idx+1)
    plt.imshow(image,cmap='gray')
    plt.title(label)
    plt.axis('off')
{% endhighlight %}

![sample images](/images/img/2017/12/cnn-sample-digits.png)


<br/><br/><br/>

## Step 1: Data and Preprocessing


The data we have contains the raw pixel values for the images we need, however in the above you notice that the array needed to be reshaped in order to display the image according to its actual visual representation.

Currently the training images exist in a 2d vector of shape 6000 * 784 however CNN's attempt to interpret not only pixel values, but the spacial relationships that exist within a given image. This means that in order to make this data suitable for input to the CNN we need to input each image as a 28 * 28 dimensional array. The 28 * 28 pixels refer to the width and height (in pixels) of each image, that is the number of elements in each row and column of the matrix.

The conversion is done below.

<br/>
{% highlight ruby %}
#image data is fine but the arrays are fine but are in the wrong shape... 
#images are currently in a long row vector need to have spacial information for cnn to be effective 

print('initially shape of image data was - {}'.format(np.asarray(train_images).shape)) #60000 * 784

#first change to NP array
train_data = np.array(train_images)
test_data  = np.array(test_images)

#reshape
train_data = train_data.reshape(60000,28, 28)
test_data = test_data.reshape(10000,28, 28)

print('after re-shape its now - {}'.format(np.asarray(train_data).shape)) #60000 * 28 *28
{% endhighlight %}

* initially shape of image data was - (60000, 784)
* after re-shape its now - (60000, 28, 28)


<br/><br/>
You can see below that we are able to display the images without needing to reshape each one.
<br/><br/>
{% highlight ruby %}
#sanity check for new data
#prove it is correct shape
samples = np.random.randint(60000, size = 10)
for idx, sample in enumerate(samples):
    image = train_data[sample]
    label = train_labels[sample]
    plt.subplot(1, 10, idx+1)
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')
{% endhighlight %}

![post reshape sanity check](/images/img/2017/12/cnn-example-reshape.png)


Unfortunately this is not where our reshape drama ends. The convolutional layers take images of a particular height and width and convolve them with a model defined number of filters (kernels) of a user defined size.

Because we have more than one filter at each convolutional layer, at this stage we are essentially multiplying our (image) Matrix of 28 * 28 with a Matrix of n * 5 * 5 (using 5 as a typical filter size).

As we know, we cannot multiply a 2d matrix by a 3d matrix so we need to explicitly define the dimensions of the third channel in the input image.


If our digit images were RGB format, and had dimensions of 3 * 28 * 28, those 28 * 28 pixels would still corrispond to locations across the width and height of the image. We would also have a third parameter the '3' in this instance to refer to each colour channel. The colour channels RGB or red, green and blue will have differing intensities depending on how much of that colour is displayed at each particular pixel location.

Once we have explicitly defined our matrix dimensions they will suitable for convolution in our model. Convolution involves taking each filter and placing it at every spacial location of the input image, in order to compute the product itself and each of the pixels that it overlays. This is done at each spacial location across all chanels of the image.

We will add the aditional dimension to our images below:

{% highlight ruby %}
{% endhighlight %}

 Now lets initialise our model ..


<br/><br/><br/>
## Step 3: Define network model (graph)


The graph has two convolutional layers separated by a ReLU activation. Following the second convolutional layer, there is a fully connected layer which is essentially a 'normal' two layer neural network.
<br/><br/>
### **Structure**

{% highlight ruby %}

class Model:
    
def __init__(self, input_dims=28, input_chnls=1, num_filters=(16, 32), filter_sz=5,
             sample_sz=2, num_classes=10, weight_scale=1e-4, initial_params=(1,2)):
        
    num_l1_filters, num_l2_filters = num_filters
    self.stride, self.pad = initial_params
        
    #initialize the conv filters
    self.c1_filters = np.random.normal(scale=weight_scale, size=(num_l1_filters, input_chnls, filter_sz, filter_sz))
    self.c1_b = np.zeros([num_l1_filters])
    self.c2_filters = np.random.normal(scale=weight_scale, size=(num_l2_filters, input_chnls, filter_sz, filter_sz))
    self.c2_b = np.zeros([num_l2_filters])
        
    #initialize the affine layer weights
    self.fc_W1 = np.random.normal(scale=weight_scale, size=(28*28 * 32, 100))
    self.fc_b1 = np.zeros([100])
    self.fc_W2 = np.random.normal(scale=weight_scale, size=(100, num_classes))
    self.fc_b2 = np.zeros([10])

{% endhighlight %}


Lets breakdown the structure based on the values above.
<br/><br/>
#### ***Convolutional Layers***
The input to the first convolution layer if we have just 2 input images is (2 * 1 * 28 * 28)
That is:
* Two images
* Each with a single colour channel (greyscale)
* With a width of size 28 and a height of size 28



Each of these images is convolved with a matrix of size (16 * 1 * 5 * 5)
That is:
* 16 Different Filters (just like we have 2 different images we have 16 different filters)
* Each with a single colour channel
* And each filter has a width of 5 pixels and a height of 5 pixels


Considering a single image for a moment this image of 1 * 28 * 28
Every channel in this image (of which there is only one) gets multiplied by 16 different filters
and the output for that filter is the summation of the product of each of the channel outputs for that particular filter.

Thus at convolutional layer 1 you have 1 * 28 * 28 at the input and 16 * 28 * 28 at the output.

At layer 2 you have 16 * 28 * 28 at the input and 32 * 28 * 28 at the output. This is because each of the 16 layers of the input is multiplied by the first of the 32 filters and the results summed to form the first of the 32 outputs. This process is repeated with each filter until all 32 filters have processed the image.

Each filter is also multiplied by a bias term.


<br/><br/>
#### ***Fully Connected Layers***
Being an 'Ordinary' Neural network the fully connected layers do not take a 2d input. 
A normal Neural Network takes its input as a long vector so we need to reshape **EACH** image into a single vector.

This vector will be the same shape that the image data was in before we rearranged it into 28 by 28 the only difference now is that for each image we have 32 input channels, therefore the length of our vector for a single image is 32 * 28 * 28.

The dimensions of the first input layer therefore are (32x28x28) * number of nodes in the hidden layer. 

As we only have a 2 layer fully connected section the dimensions of the second layer are:
(number of nodes in the hidden layer * number of output classes).

Each node in the fully connected layers is multiplied by a bias as is typical in Neural Networks.

<br/>
### **Implementation**
<br/><br/>

#### ***Forward Pass***


Let's start with the convolutional layer.



First we make a note of the padding and stride that we will be using. For simplicity these were fixed when we initialised the model.

We then find out N, how many sample images we are expecting
and H, W the height and width of those images.

We also need to find out how many Filters (or kernels) we are expecting in this layer, so k the kernels and k.shape[0] is the number of kernels we are expecting (16 in layer 1, 32 in layer 2).

HH and WW are the width and height of each kernel (indexed as the first 0th and 1st shape params of the 0th channel of the 0th kernel).

We then calculate what the output size of each image will be following convolution, this can vary depending on how big a stride is taken and whether you pad the input image or not.

we then pad the image.
Create an empty output image set ( size 16x28x28 for conv layer 1).

And **finally**
* For each image(N)
* for each pixel location in the output image (h_out and w_out). *which will equal the number of operations we need to perform on the input image*
* we will take a section of our image, whose size is equal to that of our filter, and we multiply each channel in the input image by each layer individually, sum the result and store it in the layer at the given pixel location in the output image. (once we have added the bias term.)

This is shown below

{% highlight ruby %}
def convForward(self, x, k, b):
    stride, pad = self.stride, self.pad
    N = x.shape[0]
    H, W = x[0][0].shape[0], x[0][0].shape[1]
    F = k.shape[0]
    HH, WW = k[0][0].shape[0], k[0][0].shape[1]
        
    h_out = int(1+ (H - HH + (2 * pad)) / stride )
    w_out = int(1+ (W - WW + ((2 * pad))) / stride )
        
    x_pad = np.pad((x), ( (0,), (0,), (pad,), (pad,)), 'constant')
    output = np.zeros((N, F, h_out, w_out))
    for n in range(N):
        for f in range(F):
            for h in range(h_out):
                for w in range(w_out):
                    output[n, f, h, w] = np.sum((x_pad[n, :, (h * stride) : ((h+HH)*stridew * stride) : ((w+WW)*stride)] * k[f])+ b[f])
        
    cache = (x, k, b)
    return output, cache
{% endhighlight %}

<br/><br/>

The fully connected layer is Exactly the same as we've seen in previous posts, There is a dot product of the input with the weights at layer one, plus an addition of bias terms. The result of this is passed through a ReLu activation function.

At the second layer the output from the activation function is used as to calculate the dot product with the weights at layer 2 and summed with bias terms to compute the logits (scores before the softmax is applied).



{% highlight ruby %}
def fc_forward(self, x):
    relu = lambda x: np.maximum(0, x)
            
    a1 = x.dot(self.fc_W1) + self.fc_b1
    a1_relu = relu(a1)
    scores = a1_relu.dot(self.fc_W2) + self.fc_b2 #logits
            
    return scores, (x, a1, a1_relu)
{% endhighlight %}

<br/><br/><br/>
### *Backward Pass*

As the fully connected layer is the last thing we saw in the forward pass, its the first thing we see in the backward pass. Once again for brevity I have put everything into one function.

This is essentially the same as we have seen before.
We calculate probability using the softmax and the data loss using cross entropy loss. With these values obtained we are able to figure out the derivative of each of our values with respect to this loss. That is, to what extent they contributed to the misclassification. The code is shown below.

{% highlight ruby %}
def fc_backward(self, y, y_cls, scores, cache, reg=0.05):
    x, a1, a1_relu = cache
    N = y.shape[0]
    probabilities = self.softmax(scores)
    loss = self.c_e_loss(probabilities, y_cls)
            
    data_loss = loss
    reg_loss = reg * 0.5 * np.sum(self.fc_W1*self.fc_W1) + reg * 0.5 * np.sum(self.fc_W2*self.fc_W2)
            
    loss = data_loss + reg_loss
    dscores = probabilities
    dscores[range(N),y_cls] =-1
    dscores /= N
        

    db2 = np.sum(dscores, axis=0)
    dW2 = np.dot(a1_relu.T, dscores)
            
    d_reluIn = a1_relu
    d_reluIn[a1 <= 0] = 0
        
    db1 = np.sum(d_reluIn, axis=0)
    dW1 = np.dot(x.T, d_reluIn)
    dx  = np.dot(d_reluIn, self.fc_W1.T)                   
            
    return {"b1": db1, "d2": db2, "W1": dW1, "W2": dW2, "x": dx}
{% endhighlight %}

The interesting part is what we return, we will be using our gradients for weights and biass to update W1 & W2 and b1 and b2 through stocastic gradient descent. However, the story does not end here because we still have two convolutional neural networks which we need to backpropagate through. This is why we are also returning dx.



dx, or the gradient of the loss with respect to x, is to the backpropagation layer conv2, what the loss calculated above is to fc_backward. It is the first step in calculated the respective losses of the convolutional layer 2 components. In order to make this useful we need to make a note of the shape of this matrix.

each image is currently one long 25088 (32x28x28 ) dimentional vector and needs to be reshaped in order to be compatable with the dimentionality of the layers its trying to calculate the loss of.

The reshaped (32*28*28) output can now serve as the input to the convBackward, along with the cached values from the layer 2 forward pass that we need in order to calculate our derivatives.


{% highlight ruby %}
fc_grads['x'] = fc_grads['x'].reshape(c2_out.shape)
c2_grads = self.convBackward(fc_grads['x'], c2_cache)
{% endhighlight %}



convBackward is implemented for individual layers and is called called N times depending on how many convolutional layers are present in the CNN.

The cache holds values used in the forward pass of the CNN the input, kernel and bias terms. These are needed to calculate the derivatives.

After unpacking them and noting the stride and padding used for this layer (again fixed at model initialization for simplicity), we extract the shape parameters we need for our looping constructs.

*AT THIS POINT I WOULD AGAIN LIKE TO STRESS THAT THIS FUNCTION IS MERELY ILLUSTRATIVE*
*while it will indeed work as expected the computational complexity due to all the nested loops (specifically the last one) make it horribly inefficient and unwieldy*

we calculate the loss with respect to the bias by summation as usual, however this time we sum on a filter by filter basis.

Next we recreate out initial input object which would have been input x, with padding applied. (in fairness i could have cached this and passed it in too)
we also create a placeholder for our filter values, the shape is that of the original filter matrix.

Then for each position that the filter took up in x_pad (which we assign to sub_x), our loss is the accumulation of the loss at each of these positions with respect to the output x.

And finally..


{% highlight ruby %}
def convBackward(self, dout, cache):
        
    x, k, b = cache
    stride, pad = self.stride, self.pad
        
    N, F, Hh, Ww = dout.shape
    N, C, H, W = x.shape
    F, C, HH, WW = k.shape
        
    db = np.zeros((F))
    for f in range(F):
        db[f] = np.sum(dout[:,f,:,:])##shape
            
            
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')
    dw = np.zeros((F, C, HH, WW))
        
    for f in range(F):
        for c in range(C):
            for h in range(HH):
                for w in range(WW):
                    sub_x = x_pad[:, c, h:(h + Hh *stride): stride, w:(w+Ww *stride):stride] #which part of X_pad did you affect
                    dw[f, c, h, w] = np.sum(dout[:,f,:,:] * sub_x)
                        


{% endhighlight %}



<br/><br/><br/>
**So how does it all come together..** 

{% highlight ruby %}
def loss(x, y, y_gt, reg, num_iterations, sample_size, learning_rate):
        
    num_samples = x.shape[0]
    num_epochs = sample_size / num_samples
        
    for it in range(num_iterations):
        batch_idx = np.random.randint(num_samples, size=sample_size)
            
        y_batch = y[batch_idx]    
        y_batch_cls = y_gt[batch_idx]
        x_batch = x[batch_idx]
        #
        # FORWARD PASS
        #
        # conv layers
        c1_out, c1_cache = self.convForward(x_batch, self.c1_filters, self.c1_b)
        c2_in = relu(c1_out)            
        c2_out, c2_cache = self.convForward(c1_in, self.c2_filters, self.c2_b)
            
        # fc layers
        # stretch out input - reshape
        fc_in_flat = c2_out.reshape(c2_out.shape[0], -1)
            
        # fc layer
        scores, fc_cache = self.fc_forward(fc_in_flat)
                 
            
        #
        # BACKWARD PASS
        #
        fc_grads =  self.fc_backward(y_batch, y_batch_cls, scores, fc_cache)
        #
        # (UN)Reshape -- change back to shape of kernels
        fc_grads['x'] = fc_grads['x'].reshape(c2_out.shape)
            
        #
        #
        c2_grads = self.convBackward(fc_grads['x'], c2_cache)
        c2_grads['x'][c1_out <= 0] = 0 #backprop Relu
        c1_grads = self.convBackward(c2_grads['x'], c1_cache)
            
            
        ##
        ## update weights and bias' here
        ##
{% endhighlight %}



