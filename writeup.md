## Project: Follow Me
____

The objective of this project was to train a fully convolutional network to perform semantic segmentation and allow for target tracking on a quadcopter.

#### Structure of the project:

All the files and their content is described below:

- [*code/model_training.ipnb*](./code/model_training.ipnb): Notebook containing the network architecture and used for training.

- [*code/Data Augmentation.ipnb*](./code/Data Augmentation.ipnb): Contains code used to perform horizontal flipping on the images and double the dataset size.

- [*data/weights/*](./data/weights/): This folder contains the weights of the final model.

- [*logs/run1.csv*](./logs/run1.csv): Contains the logs of the final runs.
____

#### Network Architecture :

lorem ipsum ..
____

#### Hyperparameters :
All hyper parameters were obtained via a manual grid search and following various recommendations found online.

- learning_rate = 0.001 : found via manual grid search. other lr used were 0.1, 0.05 and 0.0001
- batch_size = 16 : initial training was ran on a small local CPU forcing the use of small batches. After moving to the cloud instance, we found out this batch size kept performing well. Other batch size used in the grid search: 8, 32, 256.
- num_epochs = 30 :
- steps\_per\_epoch = train_size/batch_size : We follow the recommandation and adapt the number of steps per epoch to the size of the dataset and the batch size. This ensure than each sample is seen only once for a given epoch. 
- validation_steps = val_size/batch_size : We apply the same reasoning as above.
- dropout_rate=0.3 : after moving to a deeper network we increased the dropout rate to 0.6. As a consequence the network performance collapsed. Other rates selected were 0.4 and 0.2. they all led to aIoU below 0.4.
- workers = 2: the EC2 instance has two GPU. we chose this parameter to match the hardware.

- loss weights: As described above, we modified the categorical\_crossentropy to account for the severe class imbalance in the dataset. A first attempt at obtaining good weights used a inverse pixed count for each class   in a sample of 100 images in the dataset. This led to the weights: { 0.00241022, 0.14757306,  0.85001672}. This set of weight was downscaling the gradient and slowed down learning in addition to flipping the imbalance. With no gradient on misclassified background, the network was returning a large amount of false positive. We then implemented a callback to allow those weights to converge to __1__ after a predefined number of epochs. The effect on learning was catastrophic. A heuristic explanation is that by changing the weights at each epochs we were effectively changing the function to be learned by the network as it was learning. After trial and error we stopped on the weights: {1.0, 1.15, 1.2} to keep the gradient close to the baseline categorical\_crossentropy, with some added weight on the humans detection.
____

##### Concepts :

__1x1 Convolutional layers__ have the following characteristics:

*  1x1xfilter_size (HxWxD),
*  stride = 1, and
*  zero (same) padding.
* 4 dimensions (batch size, height, width, depth).

Contrary to fuly connected layers, they preserve spatial information as the width x height of the parent tensor is preserved.
They are part of the decoder block of a fully convolutional network.
They should be used when we wish to preserve spatial information in the network.
In addition they allow us to feed images of different size inside the network.
To use them we just need to set the parameter of our conv2d layers as follow:
* 1x1 kernel size,
* stride = 1,
* same padding.
in tensorflow, the usage would be:
```python
one_conv_layer = tf.nn.conv2d(input, filter=[1,1,1,n], stride=[1,1,1,1], padding="SAME")
```
__Fully Connected layers__ are the standard layers used in a multi-layer perceptron. They are used as the final layers of standard convnets as well as most other types of neural network with 1D-2D output (Note that the second dimension would be the batch size here). They are at the basis of the universal function approximator ability of neural networks.
To use them one need to perform a matrix multiply between the input activation of the layer and the weight matrix, following by the addition of a bias.
in tensorflow, this would be:
```python
fully_connect_layer = tf.add(tf.matmul(input, weights), bias)
```
____
 
##### Image Manipulation :


Standard convnet can be summed up as an encoder followed by a series of fully connected layer. This architecture proves very useful for binary  classification tasks of the form "is there an instance of object X in the image?". But fully connected layers do not keep track of the spatial distribution of features in the image. by replacing the fully connected layer with a decoder, we can  start using deep learning to answer questions that rely on the spatial distribution of objects within the network. Pixelwise classification, also called semantic segmentation is a prime use for those encoder/decoder architecture. Some Industrial application for robotics could be :
- Identifying the exact location of an object in a cluttered environment.
- Road and pedestrian detection for self-driving vehicles.

In more details the role of the encoder is to extract features from the image. The decoder will up-scale the features generated by the encoder to map them back to a tensor of the same shape as the input image and allow pixelwise classification.

A problem that may arise is  the loss of resolution in the output tensor caused by the encoding of the features. A solution to this is the use of skip layers, where the up-sampling layers in the decoder are concatenated with the equivalent tensor in the encoder. This allows for more granular segmentation.

Another problem can be linked to the size of the network, as we may end up with  a larger number of weights than with a fully connected layer.  by using techniques such as bilinear upsampling and separable convolutional layers, we can keep the number of learnable weights manageable. 
____

##### Limitation of the model and extensions :

A neural network in a supervised learning setup, such as the one in this project, can only learn a function based on the input it was provided with.
Our solution uses a dataset composed only of humans, with the target individual a human with very specific features (woman, with brown hair, white skin and reddish clothes). The network can only learn to identify those features as it was not presented with anything else. This means that the network we've trained can only be applied to following this specific woman and identifying the presence of other humans in the field of frame. 

To follow other target types, such as animals or other specific humains, those would need to be integrated to the training data and the model would need to be trained again. 
To be able to follow several target with the same network, the output layer depth should be expanded for each additional potential target. At deployment time, the navigation node of the robot/quadcopter could use the appropriate the network to obtain the pixels classified with its chosen target. 
____
#####  Further improvements:

At this stage, we believe that the best way to improve the model is to collect more data. Due to the memory limitation of our machine, the simulator kept crashing when recording images for more than a few seconds. Therefore we had to find other ways to increase the size of the dataset. We resorted to a simple data augmentation via horizontal flipping. After more research, it appears that several other image augmentation techniques have been developped by the computer vision community. The following Kaggle [*kernel*](https://www.kaggle.com/gaborfodor/augmentation-methods) displays a few of them. 

We noticed that the model had it worst performance when identifying the target from afar. If we can collect more data from the simulator, our main focus should be on finding more samples with this use case.