---
layout: post
title:  "CNN in TensorFlow - CIFAR-10 Classification"
date:   2017-12-31 10:00:24 +0000
category: "Neural Networks"
---

<br/><br/>
As I've spent quite a bit of time demonstrating raw convnets on the MNIST dataset. I thought I'd mix things up quickly and wizz through an implementation of a 2 layer convolutional neural network and train it on  the CIFAR-10 dataset.

Details of the CIFAR-10 dataset can be found HERE.

The dataset is based on this paper:   
***Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009***

Essentially its a dataset of 60000 32x32 training images in 10 classes, 6000 images per class.
The classes are:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck


we'll now see how simple it is to implement a convolutional neural network in tensorflow.
I'll also be adding some additional layers to the network in order to help learning to converge quicker as well as improve the accuracy of predictions.

<br/><br/>
# 1. Import Data


    #Import training data
    data_all = [None]*2
    data = []
    labels = []
    for i in range(1,6):
        with open("cs231n/datasets/cifar-10-batches-py/data_batch_%d" % i, 'rb') as db:
            data_all = pickle.load(db, encoding='bytes')
            data.extend(data_all[b'data'].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8"))
            labels.extend(data_all[b'labels'])

    data = np.array(data)
    labels = np.array(labels)
    print(data.shape) 

    #Sanity Check.. Visualise data
    for i in range(5):
        image = data[i]
        plt.subplot(1,5,i+1)
        plt.imshow(image)    


<br/>
![train images](/images/img/2017/12/cnn-tf-cifar-1.png)
<br/><br/>

    #import test data
    test_data = []
    test_labels = []
    with open("cs231n/datasets/cifar-10-batches-py/data_batch_%d" % i, 'rb') as db:
        td = pickle.load(db, encoding='latin1')
        test_data.extend(td['data'].reshape(10000, 3, 32, 32).transpose(    0,2,3,1).astype("uint8"))
        test_labels.extend(td['labels'])

    #Sanity Check.. Visualise data
    for i in range(5):
        image = test_data[i]
        plt.subplot(1,5,i+1)
        plt.imshow(image)

<br/>
![test images](/images/img/2017/12/cnn-tf-cifar-2.png)

Above I am importing the Training data and test data and visualizing the corrisponding images seporately, this is because the CIFAR-10 dataset comes pre-seporated therefore the user is not required to bulk import and manually partition the data.

The images can be seen below, they are random samples from 5 of the 10 categories. They may appear blury but thats to be expected from 32*32 pixel images!


<br/><br/>
# 2. TF-Placeholders

In tensorflow you must first define placeholders for any variables you need will be inputting into the graph from the 'outside world'. Usually this means data and labels.
Im also adding a keep_prob placeholder to allow me to perform Batch Normalization.



    # Define and train model
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32)


<br/><br/>
# 3. TF-Variables
Next you need to create your tensorflow variables which together define the componants that make up the topology of the graph.

    #
    # Create TF Variables
    #
    #Convolution layers
    wconv1 = tf.get_variable('wconv1', [5,5, 3, 32])
    bconv1 = tf.get_variable('bconv1', [32])

    #Add BatchNormalization layer between conv layers
    conv_bn_gamma = tf.get_variable('conv_bn_gamma', shape=[32])
    conv_bn_beta = tf.get_variable('conv_bn_beta', shape=[32])

    wconv2 = tf.get_variable('wconv2', [5,5,32,64])
    bconv2 = tf.get_variable('bconv2', [64])
    
    
    #affine layers
    fc_w1 = tf.get_variable('fc_w1', [16*16*64, 100])
    fc_b1 = tf.get_variable('fc_b1', [100])

    fc_w2 = tf.get_variable('fc_w2', [100, 10])
    fc_b2 = tf.get_variable('fc_b2', [10])
    
    
    #Add BatchNormalization layer between conv layers
    fc_bn_gamma = tf.get_variable('fc_bn_gamma', shape=[100])
    fc_bn_beta = tf.get_variable('fc_bn_beta', shape=[100])



<br/><br/>
# 4. Topology & Optimization
<br/>
Below we define how the layers interconnect..

    c1_out = tf.nn.conv2d(x, wconv1, strides=[1,1,1,1], padding='SAME') + bconv1
    c1_out = tf.nn.relu(c1_out)

    mean, var = tf.nn.moments(c1_out, axes=[0,1,2], keep_dims=False)
    conv_bn = tf.nn.batch_normalization(c1_out, mean, var, conv_bn_gamma, conv_bn_beta, 1e-6)

    c1_pool = tf.layers.max_pooling2d(inputs=conv_bn, pool_size=[2,2], strides=2)  


    c2_out = tf.nn.conv2d(c1_pool, wconv2, strides=[1,1,1,1], padding="SAME") + bconv2
    c2_out = tf.nn.relu(c2_out)


    c2_flat = tf.reshape(c2_out, shape=[-1, (16*16*64)])
    
    fc1_out = tf.matmul(c2_flat, fc_w1) + fc_b1
    fc1_out = tf.nn.relu(fc1_out)

    mean, var = tf.nn.moments(fc1_out, axes=[0], keep_dims=True)
    fc_bn = tf.nn.batch_normalization(fc1_out, mean, var, fc_bn_gamma, fc_bn_beta, 1e-6)


    h_fc1_drop = tf.nn.dropout(fc_bn, keep_prob)
    logits = tf.matmul(h_fc1_drop, fc_w2) + fc_b2


<br/><br/>
And let tensorflow know how we want to optimise our weights and calculate loss..

    predictions = tf.nn.softmax(logits)
    predicted_class = tf.argmax(predictions, axis=1)
    
    total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, 10), logits=logits)
    mean_loss = tf.reduce_mean(total_loss)
    
    optimizer = tf.train.AdamOptimizer(5e-4).minimize(mean_loss)
    
    correct_prediction = tf.equal(predicted_class, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


<br/><br/>
# 5. Train
For the final stop on our whistle stop tour we can run create our session and run our graph. I have chosen to run this on a GPU on a google cloud instance reduce computation time. 


    # Run using GPU
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(tf.global_variables_initializer())
            batch_size=400
           
            def train(num_its):  
                for i in range(num_its):
                    batch_idx = np.random.randint(data.shape[0], size=batch_size)
                    x_batch = data[batch_idx]
                    y_batch = labels[batch_idx]
                    training_feed_dict ={ x: x_batch, y: y_batch, keep_prob:0.5}
                    sess.run(optimizer, training_feed_dict)
                
            train(1000)
            
            test_feed_dict = {x: test_data, y: test_labels, keep_prob:1}
            acc = sess.run(accuracy, feed_dict=test_feed_dict)
            print("test set accuracy is {0:.1%}".format(acc))


